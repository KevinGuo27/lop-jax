import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
from jax_bp import Agent, compute_accuracy, ShrinkAndPerturbAgent
from jax_convnet import ConvNet
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from config import ImagenetHyperparams
import wandb
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.file_system import get_results_path, numpyify, plot_hessian_spectrum
import orbax.checkpoint
from utils.optimizer import l2_regularization, adam_with_param_counts
from utils.hessian_computation import get_hvp_fn
from utils.lanczos import lanczos_alg
from utils.density import tridiag_to_density



train_images_per_class = 600
test_images_per_class = 100
images_per_class = train_images_per_class + test_images_per_class

def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        print(f"Loading class {idx} of {len(classes)}")
        data_file = '/users/apraka15/scratch/cl/classes/' + str(_class) + '.npy'
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

def save_data(data, data_file):
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)

def repeat_expr(args: ImagenetHyperparams):
    if args.wandb:
        wandb.init(project="lop-jax", name=f"{args.agent}-{args.seed}", config=args)

    print(args)
    
    data_file = f'data/{args.agent}-{args.seed}-{args.run_idx}.pkl'
    num_epochs = args.num_showings
    classes_per_task = args.num_classes

    rng = jax.random.PRNGKey(args.seed)
    dummy_input = jnp.ones([1, 32, 32, 3])

    network = ConvNet(num_classes=args.num_classes)
    if args.agent == 'sp':
        agent = ShrinkAndPerturbAgent(network, args.shrink_factor, args.perturb_scale)
    else:
        agent = Agent(network)
    network_params = network.init(rng, dummy_input)['params']

    tx = optax.sgd(args.lr[0], args.momentum)

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    with open('class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[int(args.run_idx)]
    num_class_repetitions_required = int(args.num_classes * args.num_tasks / 1000) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)

    save_after_every_n_tasks = 1
    if args.num_tasks >= 10:
        save_after_every_n_tasks = int(args.num_tasks/10)

    examples_per_epoch = train_images_per_class * classes_per_task

    train_accuracies = np.zeros((args.num_tasks, num_epochs))
    test_accuracies = np.zeros((args.num_tasks, num_epochs))

    global_step = 0
    x_train, x_test, y_train, y_test = None, None, None, None
    for task_idx in range(args.num_tasks):
        del x_train, x_test, y_train, y_test
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx*classes_per_task:(task_idx+1)*classes_per_task])

        for epoch_idx in tqdm(range(num_epochs)):
            example_order = np.random.permutation(train_images_per_class * classes_per_task)
            x_train = x_train[example_order]
            y_train = y_train[example_order]

            new_train_accuracies = []
            for start_idx in range(0, examples_per_epoch, args.mini_batch_size):


                batch_indices = example_order[start_idx: start_idx+args.mini_batch_size]
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                batch = {'image': batch_x, 'label': batch_y}


                 #compute hessian at the start or end of each new epoch
                if args.compute_hessian and ((epoch_idx == 0 and start_idx == 0) or (epoch_idx == num_epochs - 1 and start_idx == examples_per_epoch - args.mini_batch_size)):
                    # Hessian computation on train set
                    x_hessian, y_hessian = batch_x[:args.compute_hessian_size], batch_y[:args.compute_hessian_size]
                    batch_hessian = {'image': x_hessian, 'label': y_hessian}
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

                    if start_idx == 0:
                        #fig = jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_train, density_train, task_idx, args.agent, at_init=True, to_wandb=True)
                        fig = plot_hessian_spectrum(grids_train, density_train, grids_train, density_train, task_idx, args.agent, at_init=True, to_wandb=True)
                        if args.wandb:
                            wandb.log({"hessian_spectrum_init": wandb.Image(fig)})
                            plt.close(fig)
                    else:
                        #fig = jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_train, density_train, task_idx, args.agent, at_init=False, to_wandb=True)
                        fig = plot_hessian_spectrum(grids_train, density_train, grids_train, density_train, task_idx, args.agent, at_init=False, to_wandb=True)
                        if args.wandb:
                            wandb.log({"hessian_spectrum_final": wandb.Image(fig)})
                            plt.close(fig)

                if args.agent == 'sp':
                    train_state, loss, logits, rng, grads = agent.train_step(train_state, batch, rng)
                else:
                    train_state, loss, logits, grads = agent.train_step(train_state, batch)

                if args.wandb:
                    # I want to track the gradient norms and the weight norms of the neural network in wandb. I want the l1, l2 and l_infinity norms
                    grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=1), grads)
                    grad_norms_l1 = jax.tree_util.tree_reduce(lambda x, y: x + y, grad_norms, 0)
                    grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=2), grads)
                    grad_norms_l2 = jax.tree_util.tree_reduce(lambda x, y: x + y, grad_norms, 0)
                    grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=jnp.inf), grads)
                    grad_norms_linf = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, y), grad_norms, 0)

                    weight_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=1), train_state.params)
                    weight_norms_l1 = jax.tree_util.tree_reduce(lambda x, y: x + y, weight_norms, 0)
                    weight_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=2), train_state.params)
                    weight_norms_l2 = jax.tree_util.tree_reduce(lambda x, y: x + y, weight_norms, 0)
                    weight_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=jnp.inf), train_state.params)
                    weight_norms_linf = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, y), weight_norms, 0)

                    wandb.log({
                        'grad_norm_l1': grad_norms_l1,
                        'grad_norm_l2': grad_norms_l2,
                        'grad_norm_linf': grad_norms_linf,
                        'weight_norm_l1': weight_norms_l1,
                        'weight_norm_l2': weight_norms_l2,
                        'weight_norm_linf': weight_norms_linf,
                    })
                
                new_train_accuracies.append(compute_accuracy(logits, batch_y))

            train_accuracies[task_idx][epoch_idx] = np.mean(new_train_accuracies)

            new_test_accuracies = []
            for start_idx in range(0, x_test.shape[0], args.mini_batch_size):
                test_batch_x = x_test[start_idx: start_idx + args.mini_batch_size]
                test_batch_y = y_test[start_idx: start_idx + args.mini_batch_size]
                logits = agent.predict(train_state.params, test_batch_x)
                new_test_accuracies.append(compute_accuracy(logits, test_batch_y))

            test_accuracies[task_idx][epoch_idx] = np.mean(new_test_accuracies)
            print('accuracy for task', task_idx, 'in epoch', epoch_idx, ': train, ',
                  train_accuracies[task_idx][epoch_idx], ', test,', test_accuracies[task_idx][epoch_idx])
            
            
            if args.wandb:
                wandb.log({
                    'train_accuracy': train_accuracies[task_idx][epoch_idx],
                    'test_accuracy': test_accuracies[task_idx][epoch_idx],
                    'global_step': global_step,
                    'task_idx': task_idx,
                    'epoch_idx': epoch_idx,
                })
            global_step += 1

        if task_idx % save_after_every_n_tasks == 0:
            save_data(data={
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
            }, data_file=data_file)

    save_data(data={
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    }, data_file=data_file)

    if args.wandb:
        wandb.log({
            'final_mean_train_accuracy': np.mean(train_accuracies),
            'final_mean_test_accuracy': np.mean(test_accuracies)
        })

def main():
    args = ImagenetHyperparams().parse_args()
    repeat_expr(args)


if __name__ == '__main__':
    main()

