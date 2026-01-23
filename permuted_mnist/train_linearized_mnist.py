"""
Training script for Linearized ReLU Network on Permuted MNIST.

This matches the Torch Actor structure exactly:
- fc1 (frozen): computes preactivation from raw input
- fc1_copy (trainable): initialized same as fc1, only this gets optimized
- fc_out (frozen): output layer

Forward pass (ReLU version):
    preact_init = fc1(x)  # frozen, from raw input
    relu_grad = (preact_init > 0)  # heaviside gate
    h = relu(preact_init) + relu_grad * (fc1_copy(x) - preact_init)
    out = fc_out(h)  # frozen output
"""

from collections import deque
import inspect
from time import time
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from pathlib import Path
import chex
from permuted_mnist.config import PermutedMnistHyperparams
from permuted_mnist.utils.evaluation import summarize_all_layers
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from permuted_mnist.utils.file_system import numpyify
import hashlib
import time as time_module
from permuted_mnist.utils.optimizer import adam_with_param_counts
import orbax.checkpoint


def get_results_path(args, return_npy: bool = True):
    """Get path for saving results (local version)."""
    results_dir = Path('/oscar/home/apraka15/arjun/lop-jax/permuted_mnist/results')
    results_dir.mkdir(exist_ok=True, parents=True)

    args_hash = hashlib.md5(str(args.as_dict()).encode('utf-8')).hexdigest()
    time_str = time_module.strftime("%Y%m%d-%H%M%S")

    if args.study_name is not None:
        results_dir /= args.study_name
    results_dir.mkdir(exist_ok=True, parents=True)
    results_path = results_dir / f"linearized_{args.env}_seed({args.seed})_time({time_str})_{args_hash}{'.npy' if return_npy else ''}"
    return results_path


def compute_param_norms(params):
    """Compute L1, L2, and L∞ norms of parameters"""
    flat_params = jax.tree_util.tree_leaves(params)
    flat_params = jnp.concatenate([p.flatten() for p in flat_params])

    l1_norm = jnp.sum(jnp.abs(flat_params))
    l2_norm = jnp.sqrt(jnp.sum(flat_params ** 2))
    linf_norm = jnp.max(jnp.abs(flat_params))

    return l1_norm, l2_norm, linf_norm


def compute_param_change_norms(old_params, new_params):
    """Compute L1, L2, and L∞ norms of parameter changes"""
    param_diff = jax.tree_util.tree_map(lambda x, y: x - y, new_params, old_params)
    return compute_param_norms(param_diff)


class LinearizedFFNN(nn.Module):
    """
    Single-hidden-layer linearized ReLU network matching Torch Actor structure.

    Structure:
    - fc1 (frozen): input -> hidden, computes preactivation from raw input x
    - fc1_copy (trainable): input -> hidden, initialized same as fc1
    - fc_out (frozen): hidden -> output

    Forward pass:
        preact_init = fc1(x)  # frozen
        relu_grad = (preact_init > 0)  # heaviside
        h = relu(preact_init) + relu_grad * (fc1_copy(x) - preact_init)
        out = fc_out(h)

    Only fc1_copy parameters are trainable/optimized.
    """
    num_features: int = 1000
    num_outputs: int = 10
    use_bias: bool = False  # Match Torch bias=False

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]

        # === fc1_copy (TRAINABLE) ===
        # These go in 'params' collection and will be optimized
        kernel_copy = self.param(
            'fc1_copy_kernel',
            # nn.initializers.normal(stddev=1.0 / in_features),
            # (in_features, self.num_features)
            nn.initializers.lecun_normal(),
            (in_features, self.num_features)
        )
        if self.use_bias:
            bias_copy = self.param('fc1_copy_bias', nn.initializers.zeros, (self.num_features,))
        else:
            bias_copy = None

        # === fc1 (FROZEN) ===
        # Stored in 'frozen' collection, initialized to same values as fc1_copy
        kernel_frozen = self.variable(
            'frozen', 'fc1_kernel',
            lambda: kernel_copy.copy()
        )
        if self.use_bias:
            bias_frozen = self.variable(
                'frozen', 'fc1_bias',
                lambda: bias_copy.copy()
            )
        else:
            bias_frozen = None

        # === fc_out (FROZEN) ===
        # Output layer weights, also frozen
        kernel_out = self.variable(
            'frozen', 'fc_out_kernel',
            # lambda: nn.initializers.normal(stddev=1.0 / self.num_features)(
            # (self.num_features, self.num_outputs)
            lambda: nn.initializers.lecun_normal()(
                self.make_rng('params'), (self.num_features, self.num_outputs)
            )
        )
        if self.use_bias:
            bias_out = self.variable(
                'frozen', 'fc_out_bias',
                lambda: jnp.zeros((self.num_outputs,))
            )
        else:
            bias_out = None

        # === Forward pass ===
        # Frozen preactivation from raw input x
        preact_frozen = x @ kernel_frozen.value
        if bias_frozen is not None:
            preact_frozen = preact_frozen + bias_frozen.value

        # Trainable preactivation from raw input x
        preact_copy = x @ kernel_copy
        if bias_copy is not None:
            preact_copy = preact_copy + bias_copy

        # ReLU gradient (heaviside) computed from FROZEN preactivation
        relu_grad = (preact_frozen > 0).astype(x.dtype)

        # Linearized hidden activation
        # h = relu(preact_frozen) + relu_grad * (preact_copy - preact_frozen)
        h = nn.relu(preact_frozen) + relu_grad * (preact_copy - preact_frozen)

        # Frozen output layer
        out = h @ kernel_out.value
        if bias_out is not None:
            out = out + bias_out.value

        # Return activations dict for metrics (h is the hidden layer activation)
        activations = {'layer_0': None, 'layer_1': h}

        return out, activations


class LinearizedTrainState(TrainState):
    """
    TrainState extended to hold frozen parameters.
    """
    frozen_params: Any = struct.field(pytree_node=True)


class LinearizedAgent:
    """
    Agent for linearized ReLU network training.
    """
    def __init__(self, network: LinearizedFFNN):
        self.network = network
        self.loss = jax.jit(self._loss)
        self.effective_rank_loss = jax.jit(self._effective_rank_loss)

    def predict(self, params, frozen_params, x):
        """Forward pass with trainable and frozen params."""
        output, features = self.network.apply(
            {'params': params, 'frozen': frozen_params}, x
        )
        return output, features

    def _loss(self, params, frozen_params, x, y):
        """Cross-entropy loss."""
        logits, feats = self.network.apply(
            {'params': params, 'frozen': frozen_params}, x
        )
        ce = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
        )
        return ce

    def _effective_rank_loss(self, params, frozen_params, x, er_eps=1e-8):
        """Effective rank loss computed on features (negative entropy of singular values)."""
        _, feats = self.network.apply(
            {'params': params, 'frozen': frozen_params}, x
        )
        H = feats['layer_1']  # shape [B, num_features]
        s = jnp.linalg.svdvals(H)  # [min(B, num_features)]
        s = jnp.abs(s)
        p = s / (jnp.sum(s) + er_eps)
        neg_entropy = jnp.sum(p * jnp.log(p + er_eps))  # <= 0, minimizing this maximizes effective rank
        return neg_entropy

    def effective_rank(self, features, eps=1e-8):
        """Compute effective rank of feature matrix."""
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)


def make_train(args: PermutedMnistHyperparams, rng: chex.PRNGKey):
    """
    Create the training function for linearized network.
    """
    network = LinearizedFFNN(
        num_features=args.num_features,
        num_outputs=10,  # MNIST has 10 classes
        use_bias=False,  # Match Torch bias=False
    )
    num_tasks = args.num_tasks
    images_per_class = 6000
    classes_per_task = 10
    input_size = 784
    examples_per_task = images_per_class * classes_per_task

    def train(lr, er_lr, rng):
        agent = LinearizedAgent(network)

        # Load data
        data_path = Path('/users/apraka15/arjun/lop-jax/data/mnist_')
        with open(data_path, 'rb') as f:
            x_all, y_all, _, _ = np.load(f, allow_pickle=True)
        x_all = jnp.array(x_all)
        y_all = jnp.array(y_all)

        # Initialize network
        rng, init_rng = jax.random.split(rng)
        variables = network.init({'params': init_rng}, x_all[:1])

        # Extract params (trainable fc1_copy) and frozen (fc1, fc_out)
        params = variables['params']
        frozen_params = variables['frozen']

        if args.debug:
            # Print parameter shapes to verify structure
            jax.debug.print("Trainable params (fc1_copy): {p}", p=jax.tree.map(lambda x: x.shape, params))
            jax.debug.print("Frozen params (fc1, fc_out): {p}", p=jax.tree.map(lambda x: x.shape, frozen_params))

        # Setup optimizer - only optimizes trainable params (fc1_copy)
        if args.optimizer == 'adam':
            tx = optax.chain(
                optax.add_decayed_weights(args.weight_decay),
                adam_with_param_counts(learning_rate=lr, eps=1e-5)
            )
        else:
            tx = optax.chain(
                optax.add_decayed_weights(args.weight_decay),
                optax.sgd(learning_rate=lr)
            )

        # Create train state with frozen params
        train_state = LinearizedTrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
            frozen_params=frozen_params,
        )

        def update_task(runner_state, task):
            def update_batch(runner_state, batch_idx):
                def update_step(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(
                        x, mini_batch_idx, args.mini_batch_size, axis=0
                    )
                    minibatch_y = jax.lax.dynamic_slice_in_dim(
                        y, mini_batch_idx, args.mini_batch_size, axis=0
                    )

                    # Compute loss
                    loss = agent.loss(
                        train_state.params, train_state.frozen_params,
                        minibatch_x, minibatch_y
                    )

                    # Compute accuracy
                    logits, _ = agent.predict(
                        train_state.params, train_state.frozen_params, minibatch_x
                    )
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == minibatch_y)

                    # Gradients only w.r.t. trainable params (fc1_copy)
                    grads = jax.grad(
                        lambda p: agent._loss(p, train_state.frozen_params, minibatch_x, minibatch_y)
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)

                    # Stale ER update: reuse er_grads computed on full er_batch window.
                    # Apply every minibatch step; er_step scales the update magnitude.
                    if er_grads is not None:
                        scale = -er_lr * args.er_step
                        updates = jax.tree_util.tree_map(lambda g: scale * g, er_grads)
                        new_params = optax.apply_updates(train_state.params, updates)
                        train_state = train_state.replace(params=new_params)

                    return (x, y, train_state, rng), (loss, accuracy, er_loss)

                x, y, train_state, rng = runner_state
                batch_x = jax.lax.dynamic_slice_in_dim(
                    x, batch_idx, args.mini_batch_size * args.er_batch, axis=0
                )
                batch_y = jax.lax.dynamic_slice_in_dim(
                    y, batch_idx, args.mini_batch_size * args.er_batch, axis=0
                )
                # Compute ER gradients once per er_batch window (stale for minibatches in this window).
                er_enabled = jnp.logical_and(er_lr != 0.0, args.er_step > 0)

                def compute_er(_):
                    er_loss = agent._effective_rank_loss(
                        train_state.params, train_state.frozen_params, batch_x
                    )
                    er_grads = jax.grad(
                        lambda p: agent._effective_rank_loss(p, train_state.frozen_params, batch_x)
                    )(train_state.params)
                    return er_loss, er_grads

                def no_er(_):
                    er_loss = jnp.array(0.0, dtype=batch_x.dtype)
                    er_grads = jax.tree_util.tree_map(jnp.zeros_like, train_state.params)
                    return er_loss, er_grads

                er_loss, er_grads = jax.lax.cond(er_enabled, compute_er, no_er, operand=None)
                step_runner_state = (batch_x, batch_y, train_state, rng)
                step_runner_state, (loss, accuracy, er_loss) = jax.lax.scan(
                    update_step,
                    step_runner_state,
                    jnp.arange(0, args.er_batch * args.mini_batch_size, args.mini_batch_size),
                    args.er_batch
                )
                train_state = step_runner_state[2]

                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy, er_loss)

            x, y, train_state, train_previous, rng, x_eval_set, y_eval_set = runner_state
            old_params = train_state.params

            batch_runner_state = (x, y, train_state, rng)
            batch_runner_state, (loss, accuracy, er_loss) = jax.lax.scan(
                update_batch,
                batch_runner_state,
                jnp.arange(0, examples_per_task, args.mini_batch_size * args.er_batch),
                examples_per_task // (args.mini_batch_size * args.er_batch)
            )
            accuracy = jnp.mean(accuracy)
            er_loss = jnp.mean(er_loss)
            train_state = batch_runner_state[2]
            runner_state = (x_all, y_all, train_state, rng, x_eval_set, y_eval_set)

            # Evaluate on current task (using the actual held-out eval set)
            output, features = agent.predict(
                train_state.params, train_state.frozen_params, x_eval_set
            )
            features_list = [f for f in features.values() if f is not None]
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features_list)
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_eval = jnp.mean(pred_labels == y_eval_set)

            # Evaluate on previous task's train set
            x_pretrain, y_pretrain = train_previous
            output, _ = agent.predict(
                train_state.params, train_state.frozen_params, x_pretrain
            )
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_pre = jnp.mean(pred_labels == y_pretrain)

            # Compute parameter change norms (only for trainable params)
            l1_norm_change, l2_norm_change, linf_norm_change = compute_param_change_norms(
                old_params, train_state.params
            )

            if args.debug:
                jax.debug.print(
                    "Task {t}: Train Accuracy {acc}, Eval Accuracy = {acc_eval}, "
                    "Accuracy on previous task = {acc_pretrain}",
                    t=task, acc=accuracy, acc_eval=accuracy_eval, acc_pretrain=accuracy_pre
                )
                jax.debug.print(
                    "Rank: {r}, EffRank: {er}, ApproxRank: {ar}, DeadNeurons: {dn}",
                    r=rank, er=effective_rank, ar=approx_rank, dn=dead_neurons
                )
                jax.debug.print(
                    "ER loss: {er_loss}",
                    er_loss=er_loss
                )

            res_info = {
                'loss': loss,
                'accuracy': accuracy,
                'rank': rank,
                'effective_rank': effective_rank,
                'approx_rank': approx_rank,
                'dead_neurons': dead_neurons,
                'accuracy_eval': accuracy_eval,
                'accuracy_pre': accuracy_pre,
                'er_loss': er_loss,
                'l1_norm_change': l1_norm_change,
                'l2_norm_change': l2_norm_change,
                'linf_norm_change': linf_norm_change,
            }

            return runner_state, res_info

        # Lists to collect metrics
        loss_list, acc_list, acc_pre_list, acc_eval_list = [], [], [], []
        rank_list, eff_rank_list, approx_rank_list, dead_neurons_list = [], [], [], []

        update_task_jit = jax.jit(update_task)

        # Initialize wandb if enabled
        if args.wandb:
            import wandb
            name = f"linearized_{args.lr}_{args.num_features}_{args.num_tasks}_{args.mini_batch_size}"
            wandb.init(
                project=args.wandb_project,
                name=name,
                entity=args.wandb_entity,
                group=args.wandb_group
            )
            wandb.config.update(args)

        for task in range(num_tasks):
            eval_size = args.eval_size
            train_size = examples_per_task - eval_size

            # Record previous train set
            train_previous = (x_all[train_size:], y_all[train_size:])

            # Permute dataset for current task
            rng, perm_rng = jax.random.split(rng)
            pixel_permutation = jax.random.permutation(perm_rng, input_size)
            x_all = x_all[:, pixel_permutation]

            # Shuffle data for current task
            rng, shuffle_rng = jax.random.split(rng)
            data_permutation = jax.random.permutation(shuffle_rng, examples_per_task)
            x_shuffled, y_shuffled = x_all[data_permutation], y_all[data_permutation]

            # Split into train and eval
            x_train, y_train = x_shuffled[:train_size], y_shuffled[:train_size]
            x_eval, y_eval = x_shuffled[train_size:], y_shuffled[train_size:]

            runner_state = (x_train, y_train, train_state, train_previous, rng, x_eval, y_eval)
            runner_state, res_info = update_task_jit(runner_state, task)
            _, _, train_state, rng, _, _ = runner_state[0], runner_state[1], runner_state[2], runner_state[3], runner_state[4], runner_state[5]

            # Collect metrics
            rank_list.append(res_info['rank'])
            acc_list.append(res_info['accuracy'])
            loss_list.append(res_info['loss'])
            acc_eval_list.append(res_info['accuracy_eval'])
            acc_pre_list.append(res_info['accuracy_pre'])
            eff_rank_list.append(res_info['effective_rank'])
            approx_rank_list.append(res_info['approx_rank'])
            dead_neurons_list.append(res_info['dead_neurons'])

            # Log to wandb
            if args.wandb:
                def log_to_wandb(loss, accuracy, rank, eff_rank, approx_rank, dead_neurons,
                               acc_eval, acc_pre, er_loss, l1_change, l2_change, linf_change, task_num):
                    wandb_info = {
                        'loss': float(loss),
                        'accuracy': float(accuracy),
                        'rank': float(jnp.mean(rank)),
                        'effective_rank': float(jnp.mean(eff_rank)),
                        'approx_rank': float(jnp.mean(approx_rank)),
                        'dead_neurons': float(jnp.mean(dead_neurons)),
                        'accuracy_eval': float(acc_eval),
                        'accuracy_pre': float(acc_pre),
                        'er_loss': float(er_loss),
                        'l1_norm_change': float(l1_change),
                        'l2_norm_change': float(l2_change),
                        'linf_norm_change': float(linf_change),
                        'task': int(task_num)
                    }
                    wandb.log(wandb_info)

                jax.debug.callback(
                    log_to_wandb,
                    jnp.mean(res_info['loss']), res_info['accuracy'],
                    res_info['rank'], res_info['effective_rank'],
                    res_info['approx_rank'], res_info['dead_neurons'],
                    res_info['accuracy_eval'], res_info['accuracy_pre'], res_info['er_loss'],
                    res_info['l1_norm_change'], res_info['l2_norm_change'],
                    res_info['linf_norm_change'], task
                )

        # Stack all metrics
        final_train_state = train_state
        accuracy = jnp.stack(acc_list)
        accuracy_eval = jnp.stack(acc_eval_list)
        accuracy_pre = jnp.stack(acc_pre_list)
        losses = jnp.stack(loss_list)
        eff_ranks = jnp.stack(eff_rank_list)
        dead_neurons = jnp.stack(dead_neurons_list)

        res_info = {
            'accuracy': accuracy,
            'accuracy_eval': accuracy_eval,
            'accuracy_pre': accuracy_pre,
            'loss': losses,
            'effective_rank': eff_ranks,
            'dead_neurons': dead_neurons,
            'train_state': final_train_state
        }
        return res_info

    return train


if __name__ == "__main__":
    args = PermutedMnistHyperparams().parse_args()
    print(args)
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)
    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    swept_args = deque()

    # Build vmap over all arguments (in reverse order)
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
    out = train_jit(*swept_args)
    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    final_train_state = out['train_state']

    results_path = get_results_path(args, return_npy=False)

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state
    }

    all_results = jax.tree.map(numpyify, all_results)

    # Save results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")
