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

def augment_image(image, rng):
    """Applies padding, random crop and horizontal flip."""
    # Pad the image (4 pixels on each side) → (40, 40, 3)
    padded = jnp.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
    
    # Random crop
    crop_rng, flip_rng = jax.random.split(rng)
    crop_x_rng, crop_y_rng = jax.random.split(crop_rng)
    crop_x = jax.random.randint(crop_x_rng, (), 0, 9)  
    crop_y = jax.random.randint(crop_y_rng, (), 0, 9)
    cropped = lax.dynamic_slice(padded, (crop_x, crop_y, 0), (32, 32, 3))

    # Horizontal flip
    do_flip = jax.random.bernoulli(flip_rng)
    flipped = jnp.where(do_flip, jnp.flip(cropped, axis=1), cropped)

    return flipped

batched_augment = jax.jit(jax.vmap(augment_image, in_axes=(0, 0)))

# helpers for data loading and splitting
def stratified_split(labels: np.ndarray, val_fraction: float) -> (list, list):
    """
    Perform a stratified train/validation split on an array of integer labels.

    Args:
        labels (np.ndarray of shape (N,)): integer class labels for N samples.
        val_fraction (float): fraction of each class to reserve for validation (0 <= f < 1).

    Returns:
        train_idx (list[int]): indices of samples assigned to the training set.
        val_idx   (list[int]): indices of samples assigned to the validation set.

    For each unique class in `labels`, this function:
      1. Gathers all indices of that class.
      2. Shuffles them randomly.
      3. Allocates the first `int(len(cls_inds) * val_fraction)` indices to validation,
         and the remainder to training.
    """
    train_idx, val_idx = [], []
    for cls_ in np.unique(labels):
        cls_inds = np.where(labels == cls_)[0]
        np.random.shuffle(cls_inds)
        n_val = int(len(cls_inds) * val_fraction)
        val_idx.extend(cls_inds[:n_val])
        train_idx.extend(cls_inds[n_val:])
    return train_idx, val_idx


def loader_to_arrays(loader: DataLoader):
    """
    Concatenate all batches from a DataLoader into JAX arrays.
    """
    xs, ys = [], []
    for batch in loader:
        x = batch['image']
        y = batch['label']
        xs.append(x)
        ys.append(y)
    return jnp.array(np.concatenate(xs, axis=0)), jnp.array(np.concatenate(ys, axis=0))

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

class EffectiveRankAgent:
    def __init__(self, network):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)

    def predict(self, state, x, train): # only use predict for evaluation
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

    def loss(self, state, x, y, train):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        (output, features), updates = state.apply_fn(variables, x=x, train=train, mutable=['batch_stats'])
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
        return loss, (output, features, updates)
    
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
    images_per_class = 500
    classes_per_task = 100 # since we're doing non-incremental CIFAR for now
    examples_per_task = images_per_class * classes_per_task
    all_classes = np.random.permutation(100)
    
    def make_lr_scheduler(num_tasks: int,
                        base_lr: float,
                        base_steps_per_epoch: int,
                        epochs_per_task: int,
                        drop_factor: float = 0.2,
                        drop_epochs: tuple = (60, 120, 160)):
        boundaries_and_scales = {}
        cum_steps = 0
        for t in range(1, num_tasks + 1):
            S = base_steps_per_epoch * t
            for e in drop_epochs:
                boundary = cum_steps + S * e
                boundaries_and_scales[boundary] = drop_factor
            cum_steps += epochs_per_task * S

        return optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales=boundaries_and_scales,
        )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs))
            / num_updates
        )
        return args.lr * frac
    def train(lr, er_lr, rng):
        agent = EffectiveRankAgent(network)

        # Load train, eval, and test datasets in the train function

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        data_path = Path('/users/tserapio/lop-jax/incremental_cifar_edited/data/cifar-100-python')
        dataset = CifarDataSet(
            root_dir=data_path,
            train=True,
            cifar_type=100,
            classes=None,
            device=None,
            image_normalization="max",
            label_preprocessing=None,
            use_torch=False,
            flatten=False
        )
        dataset.set_transformation(lambda sample: {
            'image': transform_train(sample['image']),
            'label': sample['label']
        })

        batch_sizes = {'train': 90, 'val': 50, 'test': 100}

        full_labels = np.array(dataset.integer_labels)
        train_idx, val_idx = stratified_split(full_labels, val_fraction=0.1)
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset,
                                  batch_size=batch_sizes['train'],
                                  shuffle=True,
                                  num_workers=4)
        val_loader   = DataLoader(val_subset,
                                  batch_size=batch_sizes['val'],
                                  shuffle=False,
                                  num_workers=4)

        test_dataset = CifarDataSet(
            root_dir=args.data_path,
            train=False,
            cifar_type=100,
            classes=None,
            device=None,
            image_normalization="max",
            label_preprocessing=None,
            use_torch=False,
            flatten=False
        )
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_sizes['test'],
                                 shuffle=False,
                                 num_workers=4)

        # data for all 100 classes already split into train/val/test
        x_train, y_train = loader_to_arrays(train_loader)
        x_train = batched_augment(x_train, jax.random.split(rng, x_train.shape[0]))
        
        x_val, y_val = loader_to_arrays(val_loader)
        
        x_test, y_test = loader_to_arrays(test_loader)
        x_all, y_all = jnp.concatenate([x_train, x_val, x_test], axis=0), jnp.concatenate([y_train, y_val, y_test], axis=0)

        # init network
        variables = network.init(rng, x_train[:1], train=True)
        network_params = variables['params']
        batch_stats = variables['batch_stats']

        lr_schedule = make_lr_scheduler(
            num_tasks = args.num_tasks, #20 for incremental cifar-100
            base_lr = args.lr, #0.1 for incremental cifar-100
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
        def update_task(runner_state, task):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)
                    
                    # compute gradients and update parameters
                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    (loss, (logits, activations, updates)), grads = grad_fn(train_state, minibatch_x, minibatch_y, train=True)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(batch_stats=updates['batch_stats'])

                    # compute accuracy 
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == minibatch_y)
                    
                    # kaicheng's old code:
                    # loss = agent.loss(train_state, minibatch_x, minibatch_y)

                    # logits, activations = agent.predict(train_state.params, minibatch_x)
                    # pred_labels = jnp.argmax(logits, axis=-1)
                    # accuracy = jnp.mean(pred_labels == minibatch_y)

                    # grads = jax.grad(agent.loss)(train_state.params, minibatch_x, minibatch_y)
                    # train_state = train_state.apply_gradients(grads=grads)
                    
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
                accuracy_runner_state, (loss, accuracy) = jax.lax.scan(update_accuracy, accuracy_runner_state, jnp.arange(0, args.er_batch * args.mini_batch_size, args.mini_batch_size), args.er_batch)
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
                    er_runner_state, er_loss = jax.lax.scan(update_erank, er_runner_state, None, args.er_step)
                    train_state = er_runner_state[1]
                
                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy)

            x, y, train_state, train_previous, rng = runner_state
            old_params = train_state.params.copy()

            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(update_erbatch, update_erbatch_runner_state, 
                                        jnp.arange(0, examples_per_task, args.mini_batch_size * args.er_batch), 
                                        examples_per_task // (args.mini_batch_size * args.er_batch))
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x_all, y_all, train_state, rng)

            # Evaluate the model on the current task
            x_eval, y_eval = x[:args.eval_size], y[:args.eval_size]
            output, features = agent.predict(train_state, x_eval, train=False)
            features_list = [f for f in features.values() if f is not None]
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features_list)
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_eval = jnp.mean(pred_labels == y_eval)

            # Evaluate the model on the previous train set
            x_pretrain, y_pretrain = train_previous
            output, features = agent.predict(train_state, x_pretrain, train=False)
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_pre = jnp.mean(pred_labels == y_pretrain)

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
            # commented out because we already loaded train/val/test 
            # eval_size = args.eval_size
            # train_size = examples_per_task - eval_size
            # # Record the previous train set
            # train_previous = (x_all[train_size:], y_all[train_size:])
            # # permuted dataset
            # rng, _rng = jax.random.split(rng)
            # pixel_permutation = jax.random.permutation(rng, input_size)
            # x_all = x_all[:, pixel_permutation]
            # # Shuffle the data for the current task
            # rng, _rng = jax.random.split(rng)
            # data_permutation = jax.random.permutation(rng, examples_per_task)
            # x_shuffled, y_shuffled = x_all[data_permutation], y_all[data_permutation]

            # # Split into train and eval sets
            # x_train, y_train = x_shuffled[:train_size], y_shuffled[:train_size]
            # x_eval, y_eval = x_shuffled[train_size:], y_shuffled[train_size:]

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
                train_previous, 
                rng)
            runner_state, res_info = update_task(runner_state, task)
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