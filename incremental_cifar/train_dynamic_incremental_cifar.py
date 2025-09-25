import sys
import json
import pickle
import argparse
from time import time
import optax
import numpy as np
from tqdm import tqdm
from incremental_cifar.model import build_resnet18
from pathlib import Path
from collections import deque
from incremental_cifar.utils.evaluation import summarize_all_layers
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import inspect
import chex
import jax
import jax.numpy as jnp
from incremental_cifar.config import IncrementalCIFARHyperparams
from incremental_cifar.utils.file_system import get_results_path, numpyify, plot_hessian_spectrum
import orbax.checkpoint
from incremental_cifar.utils.optimizer import l2_regularization, adam_with_param_counts
from incremental_cifar.utils.hessian_computation import get_hvp_fn
from incremental_cifar.utils.lanczos import lanczos_alg
from incremental_cifar.utils.density import tridiag_to_density, tridiag_to_density_and_erank
from incremental_cifar.cbp import ContinualBackpropTrainState
from typing import Any, Callable, Optional, Tuple

# create train state as we are using ResNet with BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: Any

class EffectiveRankAgent:
    def __init__(self, network):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, batch_stats, x):
        variables = {"params": params, "batch_stats": batch_stats}
        (logits_full, features), updates = self.network.apply(variables, x, train=True, mutable='batch_stats')
        features_list = [f for f in features.values() if f is not None]
        features_list = features_list[-2:] # Only take the last two layers for rank computation
        erank_losses = [self.effective_rank(f) for f in features_list]

        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank, updates

    def predict(self, params, batch_stats, x, train, active_classes):
        variables = {"params": params, "batch_stats": batch_stats}
        
        logits_full, features = self.network.apply(variables, x, train=train) # train should be False (evaluation)
        logits = logits_full[:, active_classes]

        return logits, features

    def loss(self, params, batch_stats, x, y, train, active_classes):
        variables = {"params": params, "batch_stats": batch_stats}
        (logits_full, features), updates = self.network.apply(variables, x, train=True, mutable=['batch_stats'])
        logits = logits_full[:, active_classes]
        labels_one_hot = y[:, active_classes]

        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
        return loss, (logits, features, updates)
    
    def hessian_loss(self, params, batch_stats, x, y, train, active_classes):
        # we need this because get_hvp_fn only accept a scalar loss
        variables = {"params": params, "batch_stats": batch_stats}
        (logits_full, features), updates = self.network.apply(variables, x, train=True, mutable=['batch_stats'])
        logits = logits_full[:, active_classes]
        labels_one_hot = y[:, active_classes]

        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
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

def generate_random_class_increments(num_tasks: int, total_classes: int, rng: chex.PRNGKey, num_experiments: int) -> jnp.ndarray:
    """Generate random class increments that sum to total_classes for each experiment.
    Each task gets between 2 and 8 classes.
    
    Returns:
        Array of shape (num_experiments, num_tasks) with different class increment schedules
    """
    min_per_task = 2
    max_per_task = 8
    
    # Check if it's possible to satisfy constraints
    min_total = num_tasks * min_per_task
    max_total = num_tasks * max_per_task
    
    if total_classes < min_total or total_classes > max_total:
        raise ValueError(f"Cannot distribute {total_classes} classes across {num_tasks} tasks "
                        f"with constraints [{min_per_task}, {max_per_task}] per task. "
                        f"Valid range: [{min_total}, {max_total}]")
    
    # Generate different class increment schedules for each experiment
    all_increments = []
    
    for exp_idx in range(num_experiments):
        rng, exp_key = jax.random.split(rng)
        
        # Start with minimum allocation
        increments = jnp.full(num_tasks, min_per_task)
        remaining_classes = total_classes - jnp.sum(increments)
        
        # Distribute remaining classes randomly while respecting max constraint
        for _ in range(remaining_classes):
            # Find tasks that can still accept more classes
            can_add = increments < max_per_task
            valid_indices = jnp.where(can_add, size=num_tasks, fill_value=-1)[0]
            valid_indices = valid_indices[valid_indices >= 0]  # Remove fill values
            
            if len(valid_indices) == 0:
                break  # No more tasks can accept classes
                
            # Randomly select a valid task and add one class
            exp_key, subkey = jax.random.split(exp_key)
            selected_idx = jax.random.choice(subkey, valid_indices)
            increments = increments.at[selected_idx].add(1)
        
        all_increments.append(increments)
    
    return jnp.stack(all_increments)  # Shape: (num_experiments, num_tasks)

def make_train(args: IncrementalCIFARHyperparams, rng: chex.PRNGKey):
    network = build_resnet18(num_classes=100)

    num_tasks = args.num_tasks
    train_images_per_class = 500
    test_images_per_class = 100
    images_per_class = train_images_per_class + test_images_per_class
    
    # Generate random class increments that sum to 100
    rng, increment_key = jax.random.split(rng)
    classes_per_task_array = generate_random_class_increments(num_tasks, 100, increment_key, args.num_experiments_repeat)
    
    # Debug: Print the class increments
    print(f"Random class increments for {args.num_experiments_repeat} experiments x {num_tasks} tasks:")
    for exp_idx in range(args.num_experiments_repeat):
        print(f"  Experiment {exp_idx + 1}: {classes_per_task_array[exp_idx]}")
        print(f"  Total classes: {jnp.sum(classes_per_task_array[exp_idx])} (should be 100)")
    
    num_epochs = args.num_epochs

    all_x_train, all_y_train, all_x_test, all_y_test = None, None, None, None

    with open('/users/kguo32/rl-opt/incremental_cifar/data/cifar100-onehot.pkl', 'rb') as f:
        data = pickle.load(f)

    all_x_train_np, all_y_train_np = data['x_train'], data['y_train']
    all_x_test_np,  all_y_test_np  = data['x_test'],  data['y_test']

    all_x_train = jnp.array(all_x_train_np)
    all_y_train = jnp.array(all_y_train_np)
    all_x_test = jnp.array(all_x_test_np)
    all_y_test = jnp.array(all_y_test_np)

    num_classes = 100

    rng, order_key = jax.random.split(rng)
    class_order = jax.random.permutation(order_key, num_classes)

    def train(lr, er_lr, rng):
        agent = EffectiveRankAgent(network)
        dummy_input = jnp.ones([1, 32, 32, 3])
        variables = network.init(rng, dummy_input, train=True)

        network_params = variables['params']
        batch_stats = variables['batch_stats']

        tx = optax.chain(
            optax.add_decayed_weights(args.weight_decay),
            optax.sgd(learning_rate=lr, momentum=args.momentum)
        )

        if args.cont_backprop:
            train_state = ContinualBackpropTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
                batch_stats=batch_stats,
            )
        else:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
                batch_stats=batch_stats
            )
        def update_task(runner_state, task, examples_per_epoch, active_classes, classes_this_task):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)

                    (loss, (logits, activations, updates)), grads = jax.value_and_grad(agent.loss, has_aux=True)(
                        train_state.params, train_state.batch_stats, minibatch_x, minibatch_y, True, active_classes
                    )

                    # accuracy 
                    pred_labels = jnp.argmax(logits, axis=-1)

                    labels_onehot = minibatch_y[:, active_classes]
                    true_labels  = jnp.argmax(labels_onehot, axis=-1)
                    accuracy = jnp.mean(pred_labels == true_labels)

                    # updates
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(batch_stats=updates['batch_stats'])
                    
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
                
                # ignore this
                def update_erank(runner_state, _):
                    x, train_state, rng = runner_state
                    (er_loss, batch_updates), grads = jax.value_and_grad(
                        agent.effective_rank_loss, has_aux=True
                    )(train_state.params, train_state.batch_stats, x)

                    updates = jax.tree_util.tree_map(lambda g: -er_lr * g, grads)
                    new_params = optax.apply_updates(train_state.params, updates)
                    train_state = train_state.replace(params=new_params)
                    train_state = train_state.replace(batch_stats=batch_updates["batch_stats"])
                    return (x, train_state, rng), er_loss
                        
                x, y, train_state, rng = runner_state                
                batch_x = jax.lax.dynamic_slice_in_dim(x, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                batch_y = jax.lax.dynamic_slice_in_dim(y, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                
                # ignore this first for l2 and bp, so no update_erank for now
                if args.agent in ['er', 'l2_er']:
                    er_runner_state = (batch_x, train_state, rng)
                    er_runner_state, er_loss = jax.lax.scan(
                        update_erank, 
                        er_runner_state, 
                        None, 
                        args.er_step)
                    train_state = er_runner_state[1]

                accuracy_runner_state = (batch_x, batch_y, train_state, rng)
                accuracy_runner_state, (loss, accuracy) = jax.lax.scan(
                    update_accuracy, 
                    accuracy_runner_state, 
                    jnp.arange(0, args.er_batch * args.mini_batch_size, args.mini_batch_size), 
                    args.er_batch)
                train_state = accuracy_runner_state[2]                
                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy)

            x, y, x_eval, y_eval, train_state, rng = runner_state

            active_classes = jnp.array(active_classes, dtype=jnp.int32)
            jax.debug.print("Active classes (as jnp array): {ac}", ac=active_classes)

            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(
                update_erbatch, 
                update_erbatch_runner_state, 
                jnp.arange(0, examples_per_epoch, args.mini_batch_size * args.er_batch),
                examples_per_epoch // (args.mini_batch_size * args.er_batch) # comment this out to use the full task size?? idk 
            )
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x, y, train_state, rng)

            # Evaluate the model on the current task
            logits, features = agent.predict(params=train_state.params, batch_stats=train_state.batch_stats, x=x_eval, train=False, active_classes=active_classes)
            features_list = [f for f in features.values() if f is not None]
            features_list = features_list[-2:] # Only take the last layer for rank computation
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features_list)
            
            pred_labels = jnp.argmax(logits, axis=-1)
            
            labels_onehot_eval = y_eval[:, active_classes]
            true_labels  = jnp.argmax(labels_onehot_eval, axis=-1)

            accuracy_eval = jnp.mean(pred_labels == true_labels)

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

        # Initialize containers for all experiment repetitions
        all_acc_list, all_acc_eval_list, all_eff_rank_list, all_dead_neurons_list = [], [], [], []
        
        update_task = jax.jit(update_task, static_argnums=(1,2,3,4))
        true_train_labels = jnp.argmax(all_y_train, axis=1)
        true_test_labels = jnp.argmax(all_y_test, axis=1)
        
        # Repeat the entire experiment sequence
        for experiment_idx in range(args.num_experiments_repeat):
            # Get class increments for this specific experiment
            current_classes_per_task = classes_per_task_array[experiment_idx]
            cumulative_classes = jnp.cumsum(current_classes_per_task)
            
            if args.debug:
                jax.debug.print("Starting experiment repetition {exp}/{total}", 
                               exp=experiment_idx + 1, total=args.num_experiments_repeat)
                jax.debug.print("Class increments for this experiment: {ci}", ci=current_classes_per_task)
                
            for task in range(num_tasks):
                # RESET NETWORK IF NEEDED
                if args.reset and not args.cont_backprop:
                    jax.debug.print("Resetting network after task {t}", t=task)
                    rng, _rng = jax.random.split(rng)
                    variables = network.init(rng, dummy_input, train=True)
                    network_params = variables['params']
                    batch_stats = variables['batch_stats']
                    tx = optax.chain(
                        optax.add_decayed_weights(args.weight_decay),
                        optax.sgd(learning_rate=lr, momentum=args.momentum)
                    )
                    train_state = TrainState.create(
                        apply_fn=network.apply,
                        params=network_params,
                        tx=tx,
                        batch_stats=batch_stats
                    )
                
                # Get number of classes for this specific task and cumulative classes seen so far
                classes_this_task = int(current_classes_per_task[task])
                num_classes_seen_so_far = int(cumulative_classes[task])
                
                jax.debug.print("Task {t}: Adding {ct} classes, total so far: {tot}", 
                               t=task, ct=classes_this_task, tot=num_classes_seen_so_far)
                
                examples_per_epoch = train_images_per_class * num_classes_seen_so_far

                assert (examples_per_epoch // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"

                active_classes = class_order[:num_classes_seen_so_far] # fix this to be tuple before getting passed on
                jax.debug.print("Active classes (init): {ac}", ac=active_classes)

                train_mask = jnp.isin(true_train_labels, active_classes)
                y_train = all_y_train[train_mask, :]
                x_train = jnp.transpose(jnp.compress(train_mask, all_x_train, axis=0), axes=(0, 2, 3, 1))  # transpose to (N, H, W, C)
                test_mask = jnp.isin(true_test_labels, active_classes)
                y_eval = all_y_test[test_mask, :]
                x_eval = jnp.transpose(jnp.compress(test_mask, all_x_test, axis=0), axes=(0, 2, 3, 1))  # transpose to (N, H, W, C)

                active_classes = tuple(int(x) for x in active_classes)

                #compute hessian at the start of the task
                if args.compute_hessian and task % args.compute_hessian_interval == 0:
                    # Hessian computation on test set
                    x_hessian, y_hessian = x_eval[:args.compute_hessian_size], y_eval[:args.compute_hessian_size]
                    hvp_fn, unravel, num_params = get_hvp_fn(agent.hessian_loss, train_state.params, (batch_stats, x_hessian, y_hessian, False, active_classes))
                    hvp_cl = lambda v: hvp_fn(train_state.params, v)
                    rng, _rng = jax.random.split(rng)
                    tridiag, lanczos_vecs = lanczos_alg(
                        hvp_cl,
                        num_params,
                        order=24,
                        rng_key=rng
                    )
                    # density_test, grids_test = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                    density_test, grids_test, effective_rank = tridiag_to_density_and_erank([tridiag], grid_len=10000, sigma_squared=1e-5)
                    jax.debug.print("Effective Rank at init: {er}", er=effective_rank)

                    # Hessian computation on train set
                    x_hessian, y_hessian = x_train[:args.compute_hessian_size], y_train[:args.compute_hessian_size]
                    hvp_fn, unravel, num_params = get_hvp_fn(agent.hessian_loss, train_state.params, (batch_stats, x_hessian, y_hessian, False, active_classes))
                    hvp_cl = lambda v: hvp_fn(train_state.params, v)
                    rng, _rng = jax.random.split(rng)
                    tridiag, lanczos_vecs = lanczos_alg(
                        hvp_cl,
                        num_params,
                        order=24,
                        rng_key=rng
                    )
                    density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                    jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=True, seed=args.seed)

                for epoch_idx in tqdm(range(num_epochs)):
                    rng, _rng = jax.random.split(rng)

                    runner_state = (
                        x_train,
                        y_train,
                        x_eval,
                        y_eval,
                        train_state,
                        rng)
                    runner_state, res_info = update_task(runner_state, task, examples_per_epoch, active_classes, classes_this_task)
                    x_train, y_train, train_state, rng = runner_state
                # Append directly to global flattened lists
                all_eff_rank_list.append(res_info['effective_rank'])
                all_dead_neurons_list.append(res_info['dead_neurons'])
                all_acc_list.append(res_info['accuracy'])
                all_acc_eval_list.append(res_info['accuracy_eval'])

                #compute hessian at the end of the task
                if args.compute_hessian and task % args.compute_hessian_interval == 0:
                    # TODO: Compute the Hessian
                    x_hessian, y_hessian = x_eval[:args.compute_hessian_size], y_eval[:args.compute_hessian_size]
                    hvp_fn, unravel, num_params = get_hvp_fn(agent.hessian_loss, train_state.params, (batch_stats, x_hessian, y_hessian, False, active_classes))
                    hvp_cl = lambda v: hvp_fn(train_state.params, v)
                    rng, _rng = jax.random.split(rng)
                    tridiag, lanczos_vecs = lanczos_alg(
                        hvp_cl,
                        num_params,
                        order=32,
                        rng_key=rng
                    )
                    density_test, grids_test = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

                    # Hessian computation on train set
                    x_hessian, y_hessian = x_train[:args.compute_hessian_size], y_train[:args.compute_hessian_size]
                    hvp_fn, unravel, num_params = get_hvp_fn(agent.hessian_loss, train_state.params, (batch_stats, x_hessian, y_hessian, False, active_classes))
                    hvp_cl = lambda v: hvp_fn(train_state.params, v)
                    rng, _rng = jax.random.split(rng)
                    tridiag, lanczos_vecs = lanczos_alg(
                        hvp_cl,
                        num_params,
                        order=32,
                        rng_key=rng
                    )
                    density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                    jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=False, seed=args.seed)

             
            if args.debug:
                jax.debug.print("Completed experiment repetition {exp}/{total}", 
                               exp=experiment_idx + 1, total=args.num_experiments_repeat)

        # Stack results from all experiment repetitions (flattened)
        final_train_state = runner_state[2]
        eff_ranks         = jnp.stack(all_eff_rank_list)    # Shape: (num_experiments_repeat * num_tasks,)
        dead_neurons      = jnp.stack(all_dead_neurons_list)
        acc               = jnp.stack(all_acc_list)
        acc_eval          = jnp.stack(all_acc_eval_list)

        res_info = {
            'effective_rank':  eff_ranks,
            'dead_neurons':    dead_neurons,
            'train_state':     final_train_state,
            'accuracy':        acc,
            'accuracy_eval':   acc_eval,
            'num_experiments_repeat': args.num_experiments_repeat,
        }
        return res_info
    return train

if __name__ == "__main__":
    args = IncrementalCIFARHyperparams().parse_args()
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