from collections import deque
import inspect
from time import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex
from config import PermutedMnistHyperparams
from utils.evaluation import summarize_all_layers
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from utils.file_system import get_results_path, numpyify
import orbax.checkpoint
from utils.optimizer import l2_regularization, adam_with_param_counts

# Mapping of activation names to functions
ACTIVATIONS = {
    'linear': lambda x: x,
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'selu': nn.selu,
    'swish': nn.silu,
    'leaky_relu': lambda x: nn.leaky_relu(x, negative_slope=0.01),
    'elu': nn.elu,
}

class Layer(nn.Module):
    """
    A single dense layer with an optional activation.
    """
    out_dim: int
    act_type: str = 'relu'

    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.kaiming_uniform()
        bias_init = nn.initializers.zeros
        x = nn.Dense(
            features=self.out_dim,
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=bias_init
        )(x)
        act_fn = ACTIVATIONS.get(self.act_type, lambda x: x)
        return act_fn(x)

class DeepFFNN(nn.Module):
    """
    A deep feedforward neural network with configurable depth and activations.

    Returns both the final output and a list of activations from each layer.
    """
    num_features: int = 2000
    num_outputs: int = 1
    num_hidden_layers: int = 2
    act_type: str = 'relu'

    @nn.compact
    def __call__(self, x):
        activations = []
        out = Layer(out_dim=self.num_features, act_type=self.act_type)(x)
        activations.append(out)
        for _ in range(self.num_hidden_layers - 1):
            out = Layer(out_dim=self.num_features, act_type=self.act_type)(out)
            activations.append(out)
        out = Layer(out_dim=self.num_outputs, act_type='linear')(out)
        return out, activations

class EffectiveRankAgent:
    def __init__(self, network: DeepFFNN, weight_decay: float = 0.0):
        self.network = network
        self.weight_decay = weight_decay
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def predict(self, params, x):
        output, features = self.network.apply(params, x)
        return output, features
    
    def effective_rank(self, features):
        s = jnp.linalg.svd(features, compute_uv=False)
        norm_s = s / jnp.sum(jnp.abs(s))
        entropy = -jnp.sum(jnp.where(norm_s > 0, norm_s * jnp.log(norm_s), 0.0))
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, x):
        output, features = self.network.apply(params, x)
        erank_losses = [self.effective_rank(f) for f in features]
        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank

    def loss(self, params, x, y):
        output, features = self.network.apply(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
        # l2 regularization
        if self.weight_decay > 0.0:
            loss += self.weight_decay * l2_regularization(params, alpha=self.weight_decay)
        return loss

def make_train(args: PermutedMnistHyperparams, rng: chex.PRNGKey):
    network = DeepFFNN(
        num_features=args.num_features,
        num_outputs=10,  # MNIST has 10 classes
        num_hidden_layers=args.num_hidden_layers,
        act_type=args.activation
    )
    num_tasks = args.num_tasks
    images_per_class = 6000
    classes_per_task = 10
    input_size = 784
    examples_per_task = images_per_class * classes_per_task
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs))
            / num_updates
        )
        return args.lr * frac
    def train(lr, er_lr, rng):
        if args.agent == 'er' or args.agent == 'l2_er':
            agent = EffectiveRankAgent(network, weight_decay = args.weight_decay)
        elif args.agent == 'bp':
            # TODO: Implement BP-specific training logic here
            pass
        elif args.agent == 'l2':
            # TODO: Implement L2-specific training logic here
            pass
        else:
            NotImplementedError
        
        # load data
        with open('data/mnist_', 'rb') as f:
            x_all, y_all, _, _ = np.load(f, allow_pickle=True)
        x_all = jnp.array(x_all)
        y_all = jnp.array(y_all)
        # init network
        network_params = network.init(rng, x_all[:1])

        if args.optimizer == 'sgd':
            optimizer = optax.sgd(learning_rate=lr)
        else:
            optimizer = adam_with_param_counts(learning_rate=lr, eps=1e-5)
        if args.no_anneal_lr:
            tx = optax.chain(
                # optax.clip_by_global_norm(args.max_grad_norm),
                optimizer,
            )
        else:
            tx = optax.chain(
                # optax.clip_by_global_norm(args.max_grad_norm),
                optimizer,
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        assert (examples_per_task // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"
        def update_task(runner_state, task):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, _):
                    x, y, train_state, rng = runner_state
                    loss = agent.loss(train_state.params, x, y)

                    logits, _ = agent.predict(train_state.params, x)
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == y)

                    grads = jax.grad(agent.loss)(train_state.params, x, y)
                    train_state = train_state.apply_gradients(grads=grads)
                    return (x, y, train_state, rng), (loss, accuracy)
                        
                x, y, train_state, rng = runner_state
                batch_x = jax.lax.dynamic_slice_in_dim(x, batch_idx, args.mini_batch_size, axis=0)
                batch_y = jax.lax.dynamic_slice_in_dim(y, batch_idx, args.mini_batch_size, axis=0)
                accuracy_runner_state = (batch_x, batch_y, train_state, rng)
                accuracy_runner_state, (loss, accuracy) = jax.lax.scan(update_accuracy, accuracy_runner_state, None, args.er_batch)
                train_state = accuracy_runner_state[2]
                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy)

                def update_erank(runner_state, _):
                    x, train_state, rng = runner_state
                    er_loss = agent.effective_rank_loss(train_state.params, x)
                    grads = jax.grad(agent.effective_rank_loss)(train_state.params, x)
                    train_state = train_state.apply_gradients(grads=grads)
                    return (x, train_state, rng), er_loss

                if args.agent in ['er', 'l2_er']:
                    er_runner_state = (x, train_state, rng)
                    er_runner_state, er_loss = jax.lax.scan(update_erank, er_runner_state, None, args.er_step)
                    train_state = er_runner_state[1]
                
                erbatch_runner_state = (x, y, train_state, rng)
                return erbatch_runner_state, info


            x_all, y_all, train_state, rng = runner_state
            rng, _rng = jax.random.split(rng)
            pixel_permutation = jax.random.permutation(rng, input_size)
            x_all = x_all[:, pixel_permutation]
            # Shuffle the data for the current task
            rng, _rng = jax.random.split(rng)
            data_permutation = jax.random.permutation(rng, examples_per_task)
            x, y = x_all[data_permutation], y_all[data_permutation]
            update_erbatch_runner_state = (x, y, train_state, _rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(update_erbatch, update_erbatch_runner_state, 
                                        jnp.arange(0, examples_per_task, args.mini_batch_size * args.er_batch), 
                                        examples_per_task // (args.mini_batch_size * args.er_batch))
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x_all, y_all, train_state, rng)

            # Evaluate the model on the current task
            x_eval, y_eval = x[:args.eval_size], y[:args.eval_size]
            output, features = agent.predict(train_state.params, x_eval)
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features)
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_eval = jnp.mean(pred_labels == y_eval)
            if args.debug:
                jax.debug.print("Task {t}: Train Accuracy {acc}, Eval Accuracy = {acc_eval}", t=task, acc=accuracy, acc_eval=accuracy_eval)
                jax.debug.print(
                    "Rank: {r}, EffRank: {er}, ApproxRank: {ar}, DeadNeurons: {dn}",
                    r=rank, er=effective_rank, ar=approx_rank, dn=dead_neurons
                )
                
            res_info = {
                'loss': loss,
                'accuracy': accuracy,
                'rank': rank,
                'effective_rank': effective_rank,
                'approx_rank': approx_rank,
                'dead_neurons': dead_neurons
            }
                
            return runner_state, res_info

        runner_state = (
            x_all,
            y_all,
            train_state, 
            rng)
        loss_list, acc_list, rank_list, eff_rank_list, approx_rank_list, dead_neurons_list = [], [], [], [], [], []
        update_task = jax.jit(update_task)
        for task in range(num_tasks):
            runner_state, res_info = update_task(runner_state, task)
            rank_list.append(res_info['rank'])
            eff_rank_list.append(res_info['effective_rank'])
            approx_rank_list.append(res_info['approx_rank'])
            dead_neurons_list.append(res_info['dead_neurons'])
        
        final_train_state = runner_state[2]
        ranks             = jnp.stack(rank_list)
        eff_ranks         = jnp.stack(eff_rank_list)
        approx_ranks      = jnp.stack(approx_rank_list)
        dead_neurons      = jnp.stack(dead_neurons_list)

        res_info = {
            'rank':            ranks,
            'effective_rank':  eff_ranks,
            'approx_rank':     approx_ranks,
            'dead_neurons':    dead_neurons,
            'train_state':     final_train_state
        }
        return res_info
    return train

if __name__ == "__main__":
    args = PermutedMnistHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)
    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    swept_args = deque()

    # we need to go backwards, since JAX returns indices
    # in the order in which they're vmapped.
    for i, arg in reversed(list(enumerate(train_args))):
        dims = [None] * len(train_args)
        dims[i] = 0
        vmaps_train = jax.vmap(vmaps_train, in_axes=dims)
        if arg == 'rng':
            swept_args.appendleft(rngs)
        else:
            assert hasattr(args, arg)
            swept_args.appendleft(getattr(args, arg))

    train_jit = vmaps_train
    t = time()
    print(*swept_args)
    out = train_jit(*swept_args)
    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    final_train_state = out['train_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state
    }

    all_results = jax.tree.map(numpyify, all_results)

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")