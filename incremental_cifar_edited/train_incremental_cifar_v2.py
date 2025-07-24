import sys
import json
import pickle
import argparse
from time import time
import optax
import numpy as np
from tqdm import tqdm
from modified_resnet_linen import build_resnet18
from pathlib import Path
from collections import deque
from utils.evaluation import summarize_all_layers
import optax
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

# necessary mlproj-manager imports
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
from torch.utils.data import DataLoader, Subset

# data loading helpers
def stratified_split(labels: np.ndarray, val_fraction: float) -> (list, list):
    train_idx, val_idx = [], []
    for cls_ in np.unique(labels):
        cls_inds = np.where(labels == cls_)[0]
        np.random.shuffle(cls_inds)
        n_val = int(len(cls_inds) * val_fraction)
        val_idx.extend(cls_inds[:n_val])
        train_idx.extend(cls_inds[n_val:])
    return train_idx, val_idx


def loader_to_arrays(loader: DataLoader):
    xs, ys = [], []
    for batch in loader:
        x = batch['image']
        y = batch['label']
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            y = y.cpu().numpy()
        xs.append(x)
        ys.append(y)
    return jnp.array(np.concatenate(xs, axis=0)), jnp.array(np.concatenate(ys, axis=0))

# create train state as we are using ResNet with BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: Any

class EffectiveRankAgent:
    def __init__(self, network: ConvNet):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def predict(self, state, x, train):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        output, features = state.apply_fn(variables, x, train=train, mutable=False)
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

    def loss(self, state, x, y, train):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        (logits_full, features), updates = state.apply_fn(variables, x, train=train, mutable='batch_stats')
        logits = logits_full[:, :100] # temporary, but need to only do for seen classes
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        return loss, (logits, features, updates)
    
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

    train_data = CifarDataSet(
        root_dir=data_path, 
        train=True, 
        cifar_type=100, 
        device=None, 
        classes=None,
        image_normalization="max", 
        label_preprocessing=None, # integer labels
        use_torch=False
    )
    test_data = CifarDataSet(
        root_dir=data_path, 
        train=False,
        cifar_type=100,
        device=None,
        classes=None,
        image_normalization="max",
        label_preprocessing=None, # integer labels
        use_torch=False
    )

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    train_transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
        RandomCrop(size=32, padding=4, padding_mode="reflect"),
        RandomHorizontalFlip(p=0.5),
        RandomRotator(degrees=(0,15))
    ]
    train_data.set_transformations(train_transformations)

    test_transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std)  # center by mean and divide by std
    ]
    test_data.set_transformations(test_transformations)

    all_x_train = train_data.data["data"]
    all_y_train = train_data.data["labels"]

    all_x_test = test_data.data["data"]
    all_y_test = test_data.data["labels"]

    num_classes = 100

    rng, order_key = jax.random.split(rng)
    class_order = jax.random.permutation(order_key, num_classes)

    def load_cifar100(classes=[]):
        x_train, y_train, x_test, y_test = [], [], [], []
        classes = np.array(classes)

        for _class in classes:
            x_train.append(all_x_train[all_y_train == _class])
            y_train.append(all_y_train[all_y_train == _class])
            x_test.append(all_x_test[all_y_test == _class])
            y_test.append(all_y_test[all_y_test == _class])
        
        x_train = jnp.array(np.concatenate(x_train), dtype=jnp.float32).transpose(0, 2, 3, 1)
        y_train = jnp.array(np.concatenate(y_train))
        x_test = jnp.array(np.concatenate(x_test), dtype=jnp.float32).transpose(0, 2, 3, 1)
        y_test = jnp.array(np.concatenate(y_test))

        return x_train, y_train, x_test, y_test

    def make_lr_scheduler(num_tasks: int,
                        base_lr: float,
                        base_steps_per_epoch: int,
                        epochs_per_task: int,
                        drop_factor: float = 0.2,
                        drop_epochs: tuple = (60, 120, 160)):

        # at what steps do we have a new task? 
        task_starts = [0]
        for t in range(1, num_tasks):
            S = base_steps_per_epoch * t
            task_starts.append(task_starts[-1] + S * epochs_per_task)
        
        def lr_fn(global_step : int) -> float: 
            # which task are we on?
            t = next(i for i, start in enumerate(task_starts)
                 if i == len(task_starts)-1 or global_step < task_starts[i+1])
            
            # which epoch are we on?
            # local step within the task
            S = base_steps_per_epoch * (t+1)
            local_step = global_step - task_starts[t]
            local_epoch = local_step // S

            # piecewise constant lr schedule within the task
            if local_epoch < drop_epochs[0]:
                return base_lr 
            elif local_epoch < drop_epochs[1]:
                return base_lr * drop_factor
            elif local_epoch < drop_epochs[2]:
                return base_lr * (drop_factor**2)
            else:
                return base_lr * (drop_factor**3)

        return lr_fn

    def train(lr, er_lr, rng):
        agent = EffectiveRankAgent(network)
        dummy_input = jnp.ones([1, 32, 32, 3])
        variables = network.init(rng, dummy_input, train=True)

        network_params = variables['params']
        batch_stats = variables['batch_stats']

        lr_schedule = make_lr_scheduler(
            num_tasks = args.num_tasks, # 20 for incremental cifar-100
            base_lr = args.lr, # 0.1 for incremental cifar-100
            base_steps_per_epoch = 25,
            epochs_per_task = 200,
            drop_factor = 0.2,
            drop_epochs = (60, 120, 160),
        )
        
        if args.no_anneal_lr:
            if args.optimizer == 'adam':
                tx = optax.chain(
                    optax.add_decayed_weights(args.weight_decay),
                    adam_with_param_counts(learning_rate=lr, eps=1e-5)
                )
            else:
                tx = optax.chain(
                    optax.add_decayed_weights(args.weight_decay),
                    optax.sgd(learning_rate=lr_schedule, momentum=args.momentum)
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
        assert (examples_per_epoch // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"
        def update_task(runner_state, task):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)
                    
                    # compute gradients
                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    (loss, (logits, activations, updates)), grads = grad_fn(train_state, minibatch_x, minibatch_y, train=True)

                    logits, activations = agent.predict(train_state, minibatch_x, train=True)

                    # accuracy 
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == minibatch_y)
                    
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

            seen = class_order[: (task + 1) * classes_per_task]
            active_idx, label_map = build_lookup(seen)

            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(update_erbatch, update_erbatch_runner_state, 
                                        jnp.arange(0, examples_per_epoch, args.mini_batch_size * args.er_batch), 
                                        examples_per_epoch // (args.mini_batch_size * args.er_batch))
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x, y, train_state, rng)

            # Evaluate the model on the current task
            output_full, features = agent.predict(train_state, x_eval, train=False)
            output = output_full[:, active_idx]
            y_local = label_map[y_eval]

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
            x_train, y_train, x_eval, y_eval = load_cifar100(class_order[: (task + 1) * classes_per_task]) # load cumulative data for seen classes
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
                example_order = jax.random.permutation(rng, train_images_per_class * classes_per_task) # shuffles the train data
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