import sys
import json
import pickle
import argparse
from time import time
import optax
import numpy as np
from tqdm import tqdm
from imagenet.models import ConvNet
from pathlib import Path
from collections import deque
from imagenet.utils.evaluation import summarize_all_layers
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import inspect
import chex
import jax
import jax.numpy as jnp
from imagenet.config import ImagenetHyperparams
from imagenet.utils.file_system import get_results_path, numpyify, plot_hessian_spectrum
import orbax.checkpoint
from imagenet.utils.optimizer import l2_regularization, adam_with_param_counts
from imagenet.utils.hessian_computation import get_hvp_fn
from imagenet.utils.lanczos import lanczos_alg
from imagenet.utils.density import tridiag_to_density, tridiag_to_density_and_erank
from imagenet.cbp import ContinualBackpropTrainState

class EffectiveRankAgent:
    def __init__(self, network: ConvNet, use_spectral_reg=False, spectral_k=2, 
                 spectral_target=2.0, spectral_reg_strength=0.1, spectral_power_iter=10):
        self.network = network
        self.use_spectral_reg = use_spectral_reg
        self.spectral_k = spectral_k
        self.spectral_target = spectral_target
        self.spectral_reg_strength = spectral_reg_strength
        self.spectral_power_iter = spectral_power_iter
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def predict(self, params, x):
        output, features = self.network.apply(params, x)
        return output, features
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, x):
        output, features = self.network.apply(params, x)
        erank_losses = [self.effective_rank(f) for f in features.values() if f is not None]
        erank_losses = erank_losses[-2:]  # Only take the last two layers for rank computation
        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank

    def loss(self, params, x, y):
        output, features = self.network.apply(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
        
        if self.use_spectral_reg:
            print(f"Using Spectral Regularization: {self.use_spectral_reg}")
            spectral_reg = compute_spectral_regularization(
                params, 
                k=self.spectral_k,
                target=self.spectral_target,
                reg_strength=self.spectral_reg_strength,
                num_iter=self.spectral_power_iter
            )
            loss = loss + spectral_reg
        
        return loss
    
    def perturb(self, params, perturb_scale, rng):
        return perturb_params(params, rng, perturb_scale)

def perturb_params(params, rng, scale):
    """Add N(0, scale) noise to layer parameters (weights and biases) only."""
    
    def perturb_layer_params(layer_params, rng):
        """Perturb weights and biases of a single layer."""
        rng1, rng2 = jax.random.split(rng)
        perturbed_kernel = layer_params['kernel'] + scale * jax.random.normal(rng1, layer_params['kernel'].shape, layer_params['kernel'].dtype)
        perturbed_bias = layer_params['bias'] + scale * jax.random.normal(rng2, layer_params['bias'].shape, layer_params['bias'].dtype)
        return {'kernel': perturbed_kernel, 'bias': perturbed_bias}
    
    # Split RNG for each layer
    num_layers = len([k for k in params['params'].keys() if k.startswith('layer_')])
    rngs = jax.random.split(rng, num_layers)
    
    # Create new params dict with perturbed layer parameters
    layer_idx = 0
    for key, value in params['params'].items():
        if key.startswith('layer_'):
            params['params'][key] = perturb_layer_params(value, rngs[layer_idx])
            layer_idx += 1
        else:
            # Keep non-layer parameters unchanged
            params['params'][key] = value
    
    return params, rngs[-1]

def power_iteration(w, num_iter=10, rng_key=None):
    """Compute largest singular value using power iteration.
    
    Handles both 2D (Dense) and 4D (Conv) kernels.
    For Conv kernels, reshapes (h, w, in_ch, out_ch) -> (h*w*in_ch, out_ch).
    """
    eps = 1e-6
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Reshape Conv kernels (4D) to 2D matrices
    if len(w.shape) == 4:
        # Conv kernel: (kernel_h, kernel_w, in_channels, out_channels)
        # Reshape to: (kernel_h * kernel_w * in_channels, out_channels)
        w = w.reshape(-1, w.shape[-1])
    elif len(w.shape) != 2:
        # Skip non-2D/4D tensors (shouldn't happen for kernels, but be safe)
        return jnp.array(0.0), None, None
    
    v = jax.random.normal(rng_key, (w.shape[1], ))
    
    for j in range(num_iter):
        wv = jnp.matmul(w, v)
        v = jnp.matmul(w.T, wv)
        v = v / (jnp.linalg.norm(v) + eps)
    
    Av = jnp.matmul(w, v)
    s = jnp.linalg.norm(Av)
    u = Av / (s + eps)
    
    return s, u, v

def compute_spectral_regularization(params, k=2, target=2.0, reg_strength=0.1, num_iter=10):
    """Compute spectral regularization on network parameters."""
    reg = 0.0
    param_idx = 0
    
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        # Extract the parameter name from the path
        if len(path) >= 2:
            layer_name = path[-2].key if hasattr(path[-2], 'key') else str(path[-2])
            param_type = path[-1].key if hasattr(path[-1], 'key') else str(path[-1])
            
            if param_type == 'kernel':
                # Use deterministic key based on parameter index to avoid randomness in gradients
                rng_key = jax.random.PRNGKey(param_idx)
                largest_sv, _, _ = power_iteration(leaf, num_iter=num_iter, rng_key=rng_key)
                spectral_reg = (largest_sv**k - target)**2
                reg += reg_strength * spectral_reg
                param_idx += 1
            elif param_type == 'bias':
                reg += reg_strength * jnp.sum((leaf - 0.0)**2)
            elif param_type == 'scale':
                reg += reg_strength * jnp.sum((leaf - 1.0)**2)
    
    return reg

def make_train(args: ImagenetHyperparams, rng: chex.PRNGKey):
    network = ConvNet(use_layernorm=args.use_layernorm)
    print(f"Using LayerNorm: {args.use_layernorm}")
    num_tasks = args.num_tasks
    train_images_per_class = 600
    test_images_per_class = 100
    images_per_class = train_images_per_class + test_images_per_class
    classes_per_task = 2
    examples_per_epoch = train_images_per_class * classes_per_task
    num_epochs = args.num_epochs
    
    def load_imagenet(classes=[]):
        x_train, y_train, x_test, y_test = [], [], [], []
        for idx, _class in enumerate(classes):
            print(f"Loading class {idx} of {len(classes)}")
            data_file = '/users/kguo32/data/kguo32/lop/imagenet/data/classes/' + str(_class) + '.npy'
            new_x = np.load(data_file)
            x_train.append(new_x[:train_images_per_class])
            x_test.append(new_x[train_images_per_class:])
            y_train.append(np.array([idx] * train_images_per_class))
            y_test.append(np.array([idx] * test_images_per_class))
        x_train = jnp.array(np.concatenate(x_train), dtype=jnp.float32).transpose(0, 2, 3, 1)
        y_train = jnp.array(np.concatenate(y_train))
        x_test = jnp.array(np.concatenate(x_test), dtype=jnp.float32).transpose(0, 2, 3, 1)
        y_test = jnp.array(np.concatenate(y_test))
        return x_train, y_train, x_test, y_test

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs))
            / num_updates
        )
        return args.lr * frac
    with open('/users/kguo32/rl-opt/imagenet/class_order', 'rb+') as f:
        class_order = pickle.load(f)
        # randomly choose a number from 0-300
        rng, _rng = jax.random.split(rng)
        class_idx = jax.random.randint(_rng, (), 0, len(class_order))
        class_order = class_order[int(class_idx)]

    def train(lr, er_lr, rng):
        agent = EffectiveRankAgent(
            network,
            use_spectral_reg=args.use_spectral_reg,
            spectral_k=args.spectral_k,
            spectral_target=args.spectral_target,
            spectral_reg_strength=args.spectral_reg_strength,
            spectral_power_iter=args.spectral_power_iter
        )
        dummy_input = jnp.ones([1, 32, 32, 3])
        network_params = network.init(rng, dummy_input)
        if args.no_anneal_lr:
            if args.optimizer == 'adam':
                tx = optax.chain(
                    optax.add_decayed_weights(args.weight_decay),
                    adam_with_param_counts(learning_rate=lr, eps=1e-5)
                )
            else:
                tx = optax.chain(
                    optax.add_decayed_weights(args.weight_decay),
                    optax.sgd(learning_rate=lr, momentum=args.momentum)
                )
        if args.cont_backprop:
            train_state = ContinualBackpropTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        assert (examples_per_epoch // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"
        def update_task(runner_state, task):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)
                    loss = agent.loss(train_state.params, minibatch_x, minibatch_y)

                    logits, activations = agent.predict(train_state.params, minibatch_x)
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == minibatch_y)

                    grads = jax.grad(agent.loss)(train_state.params, minibatch_x, minibatch_y)
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    if args.to_perturb:
                        rng, _rng = jax.random.split(rng)
                        new_params, rng = agent.perturb(train_state.params, args.perturb_scale, _rng)
                        train_state = train_state.replace(params=new_params)
                    if args.cont_backprop:
                        rng, _rng = jax.random.split(rng)
                        train_state = train_state.update_and_reinit(_rng,
                                                                    activations,
                                                                    replacement_rate=args.replacement_rate,
                                                                    decay_rate=args.decay_rate,
                                                                    maturity_threshold=args.maturity_threshold)
                    return (x, y, train_state, rng), (loss, accuracy)
                
                def update_erank(runner_state, _):
                    x, train_state, rng = runner_state
                    er_loss = agent.effective_rank_loss(train_state.params, x)
                    grads = jax.grad(agent.effective_rank_loss)(train_state.params, x)
                    updates = jax.tree_util.tree_map(lambda g: -er_lr * g, grads)
                    new_params = optax.apply_updates(train_state.params, updates)
                    train_state = train_state.replace(params=new_params)
                    return (x, train_state, rng), er_loss
                        
                x, y, train_state, rng = runner_state                
                batch_x = jax.lax.dynamic_slice_in_dim(x, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                batch_y = jax.lax.dynamic_slice_in_dim(y, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                if args.agent in ['er', 'l2_er']:
                    er_runner_state = (batch_x, train_state, rng)
                    er_runner_state, er_loss = jax.lax.scan(update_erank, er_runner_state, None, args.er_step)
                    train_state = er_runner_state[1]
                accuracy_runner_state = (batch_x, batch_y, train_state, rng)
                accuracy_runner_state, (loss, accuracy) = jax.lax.scan(update_accuracy, accuracy_runner_state, jnp.arange(0, args.er_batch * args.mini_batch_size, args.mini_batch_size), args.er_batch)
                train_state = accuracy_runner_state[2]                
                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy)

            x, y, x_eval, y_eval, train_state, rng = runner_state

            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(update_erbatch, update_erbatch_runner_state, 
                                        jnp.arange(0, examples_per_epoch, args.mini_batch_size * args.er_batch), 
                                        examples_per_epoch // (args.mini_batch_size * args.er_batch))
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x, y, train_state, rng)

            # Evaluate the model on the current task
            output, features = agent.predict(train_state.params, x_eval)
            features_list = [f for f in features.values() if f is not None]
            features_list = features_list[-2:] # Only take the last two layers for rank computation
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features_list)
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
                'dead_neurons': dead_neurons,
                'accuracy_eval': accuracy_eval,
            }
                
            return runner_state, res_info

        loss_list, acc_list, acc_eval_list, rank_list, eff_rank_list, approx_rank_list, dead_neurons_list = [], [], [], [], [], [], []
        update_task = jax.jit(update_task)
        for task in range(num_tasks):
            x_train, y_train, x_eval, y_eval = load_imagenet(class_order[task*classes_per_task:(task+1)*classes_per_task])
            #compute hessian at the start of the task
            if args.compute_hessian and task % args.compute_hessian_interval == 0:
                # Hessian computation on test set
                x_hessian, y_hessian = x_eval[:args.compute_hessian_size], y_eval[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                # density_test, grids_test = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                density_test, grids_test, effective_rank = tridiag_to_density_and_erank([tridiag], grid_len=10000, sigma_squared=1e-5)
                jax.debug.print("Effective Rank at init: {er}", er=effective_rank)

                # Hessian computation on train set
                x_hessian, y_hessian = x_train[:args.compute_hessian_size], y_train[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=True, seed=args.seed)

            for epoch_idx in tqdm(range(num_epochs)):
                rng, _rng = jax.random.split(rng)
                example_order = jax.random.permutation(rng, train_images_per_class * classes_per_task)
                x_train = x_train[example_order]
                y_train = y_train[example_order]
                runner_state = (
                    x_train,
                    y_train,
                    x_eval,
                    y_eval,
                    train_state,
                    rng)
                runner_state, res_info = update_task(runner_state, task)
                x_train, y_train, train_state, rng = runner_state
            rank_list.append(res_info['rank'])
            eff_rank_list.append(res_info['effective_rank'])
            approx_rank_list.append(res_info['approx_rank'])
            dead_neurons_list.append(res_info['dead_neurons'])
            acc_list.append(res_info['accuracy'])
            acc_eval_list.append(res_info['accuracy_eval'])
            loss_list.append(res_info['loss'])

            #compute hessian at the end of the task
            if args.compute_hessian and task % args.compute_hessian_interval == 0:
                # TODO: Compute the Hessian
                x_hessian, y_hessian = x_eval[:args.compute_hessian_size], y_eval[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_test, grids_test = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

                # Hessian computation on train set
                x_hessian, y_hessian = x_train[:args.compute_hessian_size], y_train[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=False, seed=args.seed)

        final_train_state = runner_state[2]
        ranks             = jnp.stack(rank_list)
        eff_ranks         = jnp.stack(eff_rank_list)
        approx_ranks      = jnp.stack(approx_rank_list)
        dead_neurons      = jnp.stack(dead_neurons_list)
        acc               = jnp.stack(acc_list)
        acc_eval          = jnp.stack(acc_eval_list)
        loss              = jnp.stack(loss_list)

        res_info = {
            'rank':            ranks,
            'effective_rank':  eff_ranks,
            'approx_rank':     approx_ranks,
            'dead_neurons':    dead_neurons,
            'train_state':     final_train_state,
            'accuracy':        acc,
            'accuracy_eval':   acc_eval,
            'loss':            loss,
        }
        return res_info
    return train

if __name__ == "__main__":
    args = ImagenetHyperparams().parse_args()
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
