from collections import deque
import inspect
from time import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from pathlib import Path
import chex
from config import IncrementalCIFARHyperparams
from utils.evaluation import summarize_all_layers
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from utils.file_system import get_results_path, numpyify, plot_hessian_spectrum
import orbax.checkpoint
from utils.optimizer import l2_regularization, adam_with_param_counts
from utils.hessian_computation import get_hvp_fn
from utils.lanczos import lanczos_alg
# from definitions import ROOT_DIR
from utils.density import tridiag_to_density, tridiag_to_density_and_erank
from cbp import ContinualBackpropTrainState

from modified_resnet_linen import build_resnet18
from typing import Any

from mlproj_manager.problems import CifarDataSet
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# create train state as we are using ResNet with BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: Any

# helpers for data loading and splitting
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
    """Concatenate all batches from a DataLoader into JAX arrays. """
    xs, ys = [], []
    for batch in loader:
        x = batch['image']
        y = batch['label']
        xs.append(x)
        ys.append(y)
    return jnp.array(np.concatenate(xs, axis=0)), jnp.array(np.concatenate(ys, axis=0))

def load_task_data(data_path, class_orde2r, task, classes_per_task, transform_train, transform_test,
                   batch_sizes, val_fraction):
    """
    Returns JAX arrays (x_train, y_train, x_val, y_val, x_test, y_test) 
    for all classes seen up through `task`.
    """
    # 1) Determine which classes we’ve “activated” by this task
    num_seen       = (task + 1) * classes_per_task
    active_classes = class_order[:num_seen].tolist()

    # 2) Build a train+val dataset just on those classes
    train_ds = CifarDataSet(
        root_dir=data_path,
        train=True,
        cifar_type=100,
        classes=active_classes,        # ← only these
        use_torch=False,
        image_normalization="max",
        label_preprocessing=None,
        flatten=False,
        device=None
    )
    train_ds.set_transformation(lambda s: {
        'image': transform_train(s['image']),
        'label': s['label']
    })

    # stratified split
    labels = np.array(train_ds.integer_labels)
    tr_idx, val_idx = stratified_split(labels, val_fraction)

    tr_loader  = DataLoader(Subset(train_ds, tr_idx),
                            batch_size=batch_sizes['train'],
                            shuffle=True,  num_workers=4)
    val_loader = DataLoader(Subset(train_ds, val_idx),
                            batch_size=batch_sizes['val'],
                            shuffle=False, num_workers=4)

    # turn them into arrays for your scan
    x_train, y_train = loader_to_arrays(tr_loader)
    x_val,   y_val   = loader_to_arrays(val_loader)

    # 3) Build a test set on the same active classes
    test_ds = CifarDataSet(
        root_dir=data_path,
        train=False,
        cifar_type=100,
        classes=active_classes,        # ← only these
        use_torch=False,
        image_normalization="max",
        label_preprocessing=None,
        flatten=False,
        device=None
    )
    test_ds.set_transformation(lambda s: {
        'image': transform_test(s['image']),
        'label': s['label']
    })
    test_loader = DataLoader(test_ds,
                             batch_size=batch_sizes['test'],
                             shuffle=False, num_workers=4)
    x_test, y_test = loader_to_arrays(test_loader)

    return x_train, y_train, x_val, y_val, x_test, y_test

def compute_param_norms(params):
    """Compute L1, L2, and L∞ norms of parameters"""
    # Flatten all parameters into a single array
    flat_params = jax.tree_util.tree_leaves(params)
    flat_params = jnp.concatenate([p.flatten() for p in flat_params])
    
    l1_norm = jnp.sum(jnp.abs(flat_params))
    l2_norm = jnp.sqrt(jnp.sum(flat_params ** 2))
    linf_norm = jnp.max(jnp.abs(flat_params))
    
    return l1_norm, l2_norm, linf_norm

def compute_param_change_norms(old_params, new_params):
    """Compute L1, L2, and L∞ norms of parameter changes"""
    # Compute parameter differences
    param_diff = jax.tree_util.tree_map(lambda x, y: x - y, new_params, old_params)
    return compute_param_norms(param_diff)

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

# use only predict for evaluation, train with loss function 
# also, don't forget about slicing the output of the model to only the current classes
class EffectiveRankAgent:
    def __init__(self, network):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)

    def predict(self, state, x, train): # only use predict for evaluation
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        output, features = state.apply_fn(variables, x=x, train=train)
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
        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank

    def loss(self, state, x, y, active_classes, train):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        (output_full, features), updates = state.apply_fn(variables, x=x, train=train, mutable=['batch_stats'])
        
        K = len(active_classes)
        output = output_full[:, active_classes]  # consider only the active classes

        num_total = output_full.shape[1] # should be 100 for CIFAR-100
        class_to_idx = jnp.full((num_total,), -1, dtype=jnp.int32)
        class_to_idx = class_to_idx.at[active_classes].set(jnp.arange(K, dtype=jnp.int32))

        y_mapped = class_to_idx[y]

        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=output, 
                labels=y_mapped
            )
        )
        return loss, (output, features, updates, y_mapped)
    
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
    num_tasks = args.num_tasks # this should be 20
    images_per_class = 500 # 450 for training, 50 for validation 
    classes_per_task = 5 # since we're doing incremental CIFAR
    examples_per_task = images_per_class * classes_per_task # 2500 (2250 training, 250 validation)
    class_order = np.random.permutation(100)
    
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

        # Load train, eval, and test datasets in the train function= 
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test  = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])  # just normalization

        batch_sizes = {'train': 90, 'val': 50, 'test': 100}
        val_fraction = 0.1
        data_path = Path('/users/tserapio/lop-jax/incremental_cifar_edited/data/cifar-100-python')

        # init network
        dummy_input = jnp.zeros((1, 3, 32, 32))
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
                    optax.sgd(learning_rate=lr_schedule, momentum=0.9, nesterov=False) 
                )
        if args.cont_backprop: # no need to look here first 
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
                batch_stats=batch_stats
            )

        assert (examples_per_task // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"
        def update_task(runner_state, active_classes, task):
            def update_erbatch(runner_state, batch_idx, active_classes):
                def update_accuracy(runner_state, mini_batch_idx, active_classes):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)
                    
                    # compute gradients and update parameters
                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    (loss, (logits, activations, updates, y_mapped)), grads = grad_fn(train_state, minibatch_x, minibatch_y, active_classes, train=True)

                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(batch_stats=updates['batch_stats'])

                    # compute accuracy 
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == y_mapped)
                    
                    if args.to_perturb: 
                        rng, _rng = jax.random.split(rng)
                        new_params, rng = agent.perturb(train_state.params, args.perturb_scale, _rng)
                        train_state = train_state.replace(params=new_params)
                    if args.cont_backprop: # no need to look here first
                        rng, _rng = jax.random.split(rng)
                        train_state = train_state.update_and_reinit(_rng,
                                                                    activations,
                                                                    replacement_rate=args.replacement_rate,
                                                                    decay_rate=args.decay_rate,
                                                                    maturity_threshold=args.maturity_threshold)
                    return (x, y, train_state, rng), (loss, accuracy)
                        
                x, y, train_state, rng = runner_state
                
                batch_x = jax.lax.dynamic_slice_in_dim(x, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                batch_y = jax.lax.dynamic_slice_in_dim(y, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                
                accuracy_runner_state = (batch_x, batch_y, train_state, rng)
                accuracy_runner_state, (loss, accuracy) = jax.lax.scan(
                    lambda carry, x: update_accuracy(carry, x, active_classes), # for a valid lax.scan call, need to wrap active_classes in a lambda
                    accuracy_runner_state,
                    jnp.arange(0, args.er_batch * args.mini_batch_size, args.mini_batch_size),
                    args.er_batch
                )
                train_state = accuracy_runner_state[2]

                # haven't touched this part yet
                def update_erank(runner_state, _):
                    x, train_state, rng = runner_state
                    er_loss = agent.effective_rank_loss(train_state.params, x)
                    grads = jax.grad(agent.effective_rank_loss)(train_state.params, x)
                    updates = jax.tree_util.tree_map(lambda g: -er_lr * g, grads)
                    new_params = optax.apply_updates(train_state.params, updates)
                    train_state = train_state.replace(params=new_params)
                    return (x, train_state, rng), er_loss

                if args.agent in ['er', 'l2_er']:
                    er_runner_state = (batch_x, train_state, rng)
                    er_runner_state, er_loss = jax.lax.scan(
                        update_erank, 
                        er_runner_state, 
                        None, 
                        args.er_step)
                    train_state = er_runner_state[1]
                
                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy)

            x, y, train_state, rng = runner_state
            old_params = train_state.params.copy()

            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(
                update_erbatch, 
                update_erbatch_runner_state, 
                jnp.arange(0, examples_per_task, args.mini_batch_size * args.er_batch), 
                examples_per_task // (args.mini_batch_size * args.er_batch),
                active_classes)
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            # runner_state = (x_all, y_all, train_state, rng)

            # Evaluate the model on the current task
            logits_test, _ = agent.predict(train_state, x_test, train=False)
            logits_test  = logits_test[:, active_classes]
            class_to_idx = jnp.full((100,), -1, jnp.int32).at[active_classes].set(
                  jnp.arange(active_classes.shape[0], dtype=jnp.int32))
            y_test_map   = class_to_idx[y_test]
            pred_labels = jnp.argmax(logits_test, axis=-1)
            accuracy_eval = jnp.mean(pred_labels == y_test_map)

            accuracy_pre = 0 # incremental cifar experiment doesn't do this

            l1_norm_change, l2_norm_change, linf_norm_change = compute_param_change_norms(old_params, train_state.params)

            if args.debug:
                jax.debug.print("Task {t}: Train Accuracy {acc}, Eval Accuracy = {acc_eval}, Accuracy on previous task = {acc_pretrain}", t=task, acc=accuracy, acc_eval=accuracy_eval, acc_pretrain=accuracy_pre)
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
                'accuracy_pre': accuracy_pre,
                'l1_norm_change': l1_norm_change,
                'l2_norm_change': l2_norm_change,
                'linf_norm_change': linf_norm_change
            }
                
            return runner_state, res_info

        loss_list, acc_list, acc_pre_list, acc_eval_list, rank_list, eff_rank_list, approx_rank_list, dead_neurons_list = [], [], [], [], [], [], [], []
        update_task = jax.jit(update_task)
        if args.wandb:
            import wandb
            name = f"{args.agent}_{args.activation}_{args.lr}_{args.er_lr}_{args.er_batch}_{args.er_step}_{args.num_features}_{args.num_hidden_layers}_{args.num_tasks}_{args.mini_batch_size}_{args.er_batch}_{args.er_step}"
            wandb.init(project=args.wandb_project, name=name, entity=args.wandb_entity, group=args.wandb_group)
            wandb.config.update(args)
        
        # per task 
        for task in range(num_tasks):
            num_classes_seen = (task + 1) * classes_per_task
            active_classes = class_order[:num_classes_seen].tolist()
            active_classes = jnp.array(active_classes, dtype=jnp.int32)

            x_train, y_train, x_val, y_val, x_test, y_test = load_task_data(
                data_path, class_order, task, classes_per_task, 
                transform_train, transform_test, batch_sizes, val_fraction
            )

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

            runner_state = (
                x_train,
                y_train,
                train_state,
                rng
            )
            runner_state, res_info = update_task(runner_state, active_classes, task)
            x_train, y_train, train_state, rng = runner_state
            rank_list.append(res_info['rank'])
            acc_list.append(res_info['accuracy'])
            loss_list.append(res_info['loss'])
            acc_eval_list.append(res_info['accuracy_eval'])
            acc_pre_list.append(res_info['accuracy_pre'])
            eff_rank_list.append(res_info['effective_rank'])
            approx_rank_list.append(res_info['approx_rank'])
            dead_neurons_list.append(res_info['dead_neurons'])
            
            if args.wandb:
                def log_to_wandb(loss, accuracy, rank, eff_rank, approx_rank, dead_neurons, 
                               acc_eval, acc_pre, l1_change, l2_change, linf_change, task_num):
                    wandb_info = {
                        'loss': float(loss),
                        'accuracy': float(accuracy),
                        'rank': float(jnp.mean(rank)),
                        'effective_rank': float(jnp.mean(eff_rank)),
                        'approx_rank': float(jnp.mean(approx_rank)),
                        'dead_neurons': float(jnp.mean(dead_neurons)),
                        'accuracy_eval': float(acc_eval),
                        'accuracy_pre': float(acc_pre),
                        'l1_norm_change': float(l1_change),
                        'l2_norm_change': float(l2_change),
                        'linf_norm_change': float(linf_change),
                        'task': int(task_num)
                    }
                    wandb.log(wandb_info)
                
                jax.debug.callback(log_to_wandb, 
                                 jnp.mean(res_info['loss']), res_info['accuracy'], 
                                 res_info['rank'], res_info['effective_rank'], 
                                 res_info['approx_rank'], res_info['dead_neurons'],
                                 res_info['accuracy_eval'], res_info['accuracy_pre'],
                                 res_info['l1_norm_change'], res_info['l2_norm_change'], 
                                 res_info['linf_norm_change'], task)

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
        accuracy          = jnp.stack(acc_list)
        accuracy_eval     = jnp.stack(acc_eval_list)
        accuracy_pre      = jnp.stack(acc_pre_list)
        losses            = jnp.stack(loss_list)
        ranks             = jnp.stack(rank_list)
        eff_ranks         = jnp.stack(eff_rank_list)
        approx_ranks      = jnp.stack(approx_rank_list)
        dead_neurons      = jnp.stack(dead_neurons_list)

        res_info = {
            'accuracy':        accuracy,
            'accuracy_eval':   accuracy_eval,
            'accuracy_pre':    accuracy_pre,
            'loss':            losses,
            'rank':            ranks,
            'effective_rank':  eff_ranks,
            'approx_rank':     approx_ranks,
            'dead_neurons':    dead_neurons,
            'train_state':     final_train_state
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