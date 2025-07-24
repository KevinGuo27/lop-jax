import sys
import json
import pickle
import argparse
from time import time
import optax
import numpy as np
from tqdm import tqdm
from modified_resnet_linen_dict import build_resnet18
from pathlib import Path
from collections import deque
from utils.evaluation import summarize_all_layers
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import inspect
import chex
import jax
import jax.numpy as jnp
from config import IncrementalCIFARHyperparams
from utils.file_system import get_results_path, numpyify, plot_hessian_spectrum
import orbax.checkpoint
from utils.optimizer import l2_regularization, adam_with_param_counts
from utils.hessian_computation import get_hvp_fn
from utils.lanczos import lanczos_alg
from utils.density import tridiag_to_density, tridiag_to_density_and_erank
from cbp import ContinualBackpropTrainState
from typing import Any, Callable, Optional, Tuple

# create train state as we are using ResNet with BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: Any

class EffectiveRankAgent:
    def __init__(self, network):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def predict(self, params, batch_stats, x, train):
        variables = {"params": params, "batch_stats": batch_stats}
        output, features = self.network.apply(variables, x, train=train, mutable='batch_stats')
        return output, features
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, batch_stats, x):
        variables = {"params": params, "batch_stats": batch_stats}
        output, features = self.network.apply(variables, x, train=True, mutable='batch_stats')
        erank_losses = [self.effective_rank(f) for f in features.values() if f is not None]
        erank_losses = erank_losses[-1:]  # Only take the last layer for rank computation
        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank

    def loss(self, params, batch_stats, x, y, train, active_classes):
        variables = {"params": params, "batch_stats": batch_stats}
        (logits_full, features), updates = self.network.apply(variables, x, train=train, mutable='batch_stats')
        
        logits = logits_full[:, active_classes]
        class_to_idx = jnp.full((100,), -1, dtype=jnp.int32).at[active_classes].set(jnp.arange(len(active_classes)))
        y = class_to_idx[y]

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        return loss, updates
    
    def perturb(self, params, perturb_scale, rng):
        return perturb_params(params, rng, perturb_scale)

def perturb_params(params, rng, scale):
    """Add N(0, scale) noise to every parameter tensor in the tree."""
    
    leaves, treedef = jax.tree_util.tree_flatten(params)
    rngs = jax.random.split(rng, len(leaves))
    
    new_leaves = [
        p + scale * jax.random.normal(r, p.shape, p.dtype)
        for p, r in zip(leaves, rngs)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves), rngs[-1]

def make_train(args: IncrementalCIFARHyperparams, rng: chex.PRNGKey):
    network = build_resnet18(num_classes=100)

    num_tasks = args.num_tasks
    train_images_per_class = 500
    test_images_per_class = 100
    images_per_class = train_images_per_class + test_images_per_class
    classes_per_task = 5
    examples_per_epoch = train_images_per_class * classes_per_task
    num_epochs = args.num_epochs

    all_x_train, all_y_train, all_x_test, all_y_test = None, None, None, None

    with open('./data/cifar100.pkl', 'rb') as f:
        # these are numpy arrays
        all_x_train, all_y_train, all_x_test, all_y_test = pickle.load(f)

    num_classes = 100

    rng, order_key = jax.random.split(rng)
    class_order = np.random.permutation(num_classes)

    def train(lr, er_lr, rng):
        agent = EffectiveRankAgent(network)
        dummy_input = jnp.ones([1, 32, 32, 3])
        variables = network.init(rng, dummy_input, train=True)

        network_params = variables['params']
        batch_stats = variables['batch_stats']

        tx = optax.chain(
            optax.add_decayed_weights(args.weight_decay),
            optax.sgd(learning_rate=0.1, momentum=args.momentum)
        )

        if args.cont_backprop:
            train_state = ContinualBackpropTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx
            )
        else:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
                batch_stats=batch_stats
            )
        def update_task(runner_state, task, examples_per_epoch, active_classes):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)
                    
                    loss = agent.loss(train_state.params, train_state.batch_stats, minibatch_x, minibatch_y, True, active_classes)

                    ((logits_full, activations), updates) = agent.predict(train_state.params, train_state.batch_stats, minibatch_x, True)
                    logits = logits_full[:, active_classes]

                    # accuracy 
                    pred_labels = jnp.argmax(logits, axis=-1)

                    class_to_idx = jnp.full((100,), -1, dtype=jnp.int32).at[active_classes].set(jnp.arange(len(active_classes)))
                    true_minibatch_y_labels = class_to_idx[minibatch_y]
                    accuracy = jnp.mean(pred_labels == true_minibatch_y_labels)

                    grads, updates = jax.grad(agent.loss, has_aux=True)(train_state.params, train_state.batch_stats, minibatch_x, minibatch_y, True, active_classes)

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
                    er_loss = agent.effective_rank_loss(train_state, x)
                    grads = jax.grad(agent.effective_rank_loss)(train_state, x)
                    updates = jax.tree_util.tree_map(lambda g: -er_lr * g, grads)
                    new_params = optax.apply_updates(train_state.params, updates)
                    train_state = train_state.replace(params=new_params)
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

            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(
                update_erbatch, 
                update_erbatch_runner_state, 
                jnp.arange(0, examples_per_epoch, args.mini_batch_size * args.er_batch)
                # ,examples_per_epoch // (args.mini_batch_size * args.er_batch) # remove this to use the full task size
            )
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x, y, train_state, rng)

            # Evaluate the model on the current task
            ((output_full, features), updates) = agent.predict(train_state.params, train_state.batch_stats, x_eval, train=False)
            output = output_full[:, active_classes]

            features_list = [f for f in features.values() if f is not None]
            features_list = features_list[-1:] # Only take the last layer for rank computation
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features_list)
            
            pred_labels = jnp.argmax(output, axis=-1)
            class_to_idx = jnp.full((100,), -1, dtype=jnp.int32).at[active_classes].set(jnp.arange(len(active_classes)))
            true_labels = class_to_idx[y_eval] # this remapping is necessary because model's logits are now only for the active classes

            accuracy_eval = jnp.mean(pred_labels == true_labels)

            if args.debug:
                jax.debug.print("Task {t}: Train Accuracy {acc}, Eval Accuracy = {acc_eval}", t=task, acc=accuracy, acc_eval=accuracy_eval)
                jax.debug.print(
                    "Rank: {r}, EffRank: {er}, ApproxRank: {ar}, DeadNeurons: {dn}",
                    r=rank, er=effective_rank, ar=approx_rank, dn=dead_neurons
                )
                jax.debug.print("True labels: {tl}", tl=true_labels)
                jax.debug.print("Predicted labels: {pl}", pl=pred_labels)
                jax.debug.print("Active Classes: {ac}", ac=active_classes)
                
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
        update_task = jax.jit(update_task, static_argnums=(1,2,3,))
        for task in range(num_tasks):
            num_classes_seen_so_far = (task + 1) * classes_per_task
            jax.debug.print("Task {t}: Training on classes {c}", t=task, c=num_classes_seen_so_far)
            
            examples_per_epoch = train_images_per_class * num_classes_seen_so_far

            assert (examples_per_epoch // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"

            active_classes = class_order[: (task + 1) * classes_per_task] # fix this to be tuple before getting passed on
            all_x_train, all_y_train, all_x_test, all_y_test = None, None, None, None
            
            with open('./data/cifar100.pkl', 'rb') as f:
                # these are numpy arrays
                all_x_train, all_y_train, all_x_test, all_y_test = pickle.load(f)
            
            # we need to convert to jnp arrays
            all_x_train, all_y_train, all_x_test, all_y_test = map(jnp.array, (all_x_train, all_y_train, all_x_test, all_y_test))
            
            train_mask = jnp.isin(all_y_train, active_classes)
            x_train = jnp.transpose(jnp.compress(train_mask, all_x_train, axis=0), (0, 2, 3, 1))
            y_train = jnp.compress(train_mask, all_y_train, axis=0) # axis 0 to select images

            # Similarly for test set
            test_mask = jnp.isin(all_y_test, active_classes)
            x_eval = jnp.transpose(jnp.compress(test_mask, all_x_test, axis=0), (0, 2, 3, 1))
            y_eval = jnp.compress(test_mask, all_y_test, axis=0) # axis 0 to select images

            active_classes = tuple(active_classes)  # convert to tuple for static_argnums
            
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
                jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=True)

            for epoch_idx in tqdm(range(num_epochs)):
                rng, _rng = jax.random.split(rng)
                # example_order = jax.random.permutation(rng, train_images_per_class * classes_per_task) # shuffles the train data
                # x_train = x_train[example_order]
                # y_train = y_train[example_order]

                # set the LR here by editing train_state
                if epoch_idx == 0:
                    train_state = train_state.replace(
                        tx=optax.chain(
                            optax.add_decayed_weights(args.weight_decay),
                            optax.sgd(learning_rate=0.1, momentum=args.momentum)
                        )
                    )
                elif epoch_idx % 60 == 0:
                    train_state = train_state.replace(
                        tx=optax.chain(
                            optax.add_decayed_weights(args.weight_decay),
                            optax.sgd(learning_rate=0.02, momentum=args.momentum)
                        )
                    )
                elif epoch_idx % 120 == 0:
                    train_state = train_state.replace(
                        tx=optax.chain(
                            optax.add_decayed_weights(args.weight_decay),
                            optax.sgd(learning_rate=0.004, momentum=args.momentum)
                        )
                    )
                elif epoch_idx % 160 == 0:
                    train_state = train_state.replace(
                        tx=optax.chain(
                            optax.add_decayed_weights(args.weight_decay),
                            optax.sgd(learning_rate=0.0008, momentum=args.momentum)
                        )
                    )

                # because update_task is jitted, we need to pass all_x andall_y here
                runner_state = (
                    x_train,
                    y_train,
                    x_eval,
                    y_eval,
                    train_state,
                    rng)
                runner_state, res_info = update_task(runner_state, task, examples_per_epoch, active_classes)
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
                jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=False)

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