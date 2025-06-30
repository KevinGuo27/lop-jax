import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from jax_bp import Agent, compute_accuracy
from jax_convnet import ConvNet
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

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

def repeat_expr(params: {}):
    agent_type = params['agent']
    num_tasks = params['num_tasks']
    num_showings = params['num_showings']

    step_size = params['step_size']
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'contribution'
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev='cpu'
    num_classes = 10
    total_classes = 1000
    new_heads = False
    mini_batch_size = 100
    perturb_scale = 0
    momentum = 0
    net_type = 1
    compute_hessian = False
    compute_hessian_size = 50

    if 'replacement_rate' in params.keys(): replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys(): decay_rate = params['decay_rate']
    if 'util_type' in params.keys(): util_type = params['util_type']
    if 'maturity_threshold' in params.keys():   maturity_threshold = params['maturity_threshold']
    if 'weight_decay' in params.keys(): weight_decay = params['weight_decay']       
    if 'num_classes' in params.keys():  num_classes = params['num_classes']
    if 'new_heads' in params.keys():    new_heads = params['new_heads']
    if 'mini_batch_size' in params.keys():  mini_batch_size = params['mini_batch_size']
    if 'perturb_scale' in params.keys():    perturb_scale = params['perturb_scale']
    if 'momentum' in params.keys(): momentum = params['momentum']
    if 'new_heads' in params.keys(): new_heads = params['new_heads']
    if 'net_type' in params.keys(): net_type = params['net_type']
    if 'compute_hessian' in params.keys(): compute_hessian = params['compute_hessian']
    if 'compute_hessian_size' in params.keys(): compute_hessian_size = params['compute_hessian_size']


    print(params)
    
    num_epochs = num_showings
    classes_per_task = num_classes

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones([1, 32, 32, 3])
    #state = create_train_state(rng, step_size, momentum, ConvNet, dummy_input, num_classes=num_classes)

    network = ConvNet(num_classes=num_classes)
    agent = Agent(network)
    network_params = network.init(rng, dummy_input)['params']

    tx = optax.sgd(step_size, momentum)

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    with open('class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[int([params['run_idx']][0])]
    num_class_repetitions_required = int(num_classes * num_tasks / total_classes) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)

    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    examples_per_epoch = train_images_per_class * classes_per_task

    train_accuracies = np.zeros((num_tasks, num_epochs))
    test_accuracies = np.zeros((num_tasks, num_epochs))

    x_train, x_test, y_train, y_test = None, None, None, None
    for task_idx in range(num_tasks):
        del x_train, x_test, y_train, y_test
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx*classes_per_task:(task_idx+1)*classes_per_task])

        # if new_heads:
        #     params = state.params.unfreeze()
        #     params['Dense_2']['kernel'] = jax.nn.initializers.zeros(rng, params['Dense_2']['kernel'].shape)
        #     params['Dense_2']['bias'] = jax.nn.initializers.zeros(rng, params['Dense_2']['bias'].shape)
        #     state = state.replace(params=params)


       

        for epoch_idx in tqdm(range(num_epochs)):
            example_order = np.random.permutation(train_images_per_class * classes_per_task)
            x_train = x_train[example_order]
            y_train = y_train[example_order]

            new_train_accuracies = []
            for start_idx in range(0, examples_per_epoch, mini_batch_size):


                batch_indices = example_order[start_idx: start_idx+mini_batch_size]
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                batch = {'image': batch_x, 'label': batch_y}


                 #compute hessian at the start or end of each new epoch
                if compute_hessian and ((epoch_idx == 0 and start_idx == 0) or (epoch_idx == num_epochs - 1 and start_idx == examples_per_epoch - mini_batch_size)):
                    # Hessian computation on test set
                    # Hessian computation on train set
                    x_hessian, y_hessian = batch_x[:compute_hessian_size], batch_y[:compute_hessian_size]
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
                        jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_train, density_train, task_idx, params['agent_type'], at_init=True)
                    else:
                        jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_train, density_train, task_idx, params['agent_type'], at_init=False)

            


                train_state, loss, logits = agent.train_step(train_state, batch)
                
                new_train_accuracies.append(compute_accuracy(logits, batch_y))

            train_accuracies[task_idx][epoch_idx] = np.mean(new_train_accuracies)

            new_test_accuracies = []
            for start_idx in range(0, x_test.shape[0], mini_batch_size):
                test_batch_x = x_test[start_idx: start_idx + mini_batch_size]
                test_batch_y = y_test[start_idx: start_idx + mini_batch_size]
                logits = agent.predict(train_state.params, test_batch_x)
                new_test_accuracies.append(compute_accuracy(logits, test_batch_y))

            test_accuracies[task_idx][epoch_idx] = np.mean(new_test_accuracies)
            print('accuracy for task', task_idx, 'in epoch', epoch_idx, ': train, ',
                  train_accuracies[task_idx][epoch_idx], ', test,', test_accuracies[task_idx][epoch_idx])

        if task_idx % save_after_every_n_tasks == 0:
            save_data(data={
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
            }, data_file=params['data_file'])

    save_data(data={
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    }, data_file=params['data_file'])

def main(arguments):
    # params = {
    #     'agent': 'bp',
    #     'num_tasks': 2000,
    #     'num_showings': 100,
    #     'step_size': 0.01,
    #     'opt': 'sgd',
    #     'run_idx': 0,
    #     'num_classes': 2,
    #     'data_file': 'data/imagenet_bp_jax.pkl',
    # }

    params = {
            "agent": "bp",
            "num_tasks": 2000,
            "num_classes": 2,
            "num_showings": 100, #250
            "mini_batch_size": 100,
            "opt": "sgd",
            "step_size": 0.01,
            "momentum": 0.9,
            "weight_decay": 0,
            "run_idx": 0,
            "data_file": "data/imagenet_bp_jax.pkl",
            "compute_hessian": True,
            "compute_hessian_size": 200,
            "agent_type": "bp",
        }
    repeat_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
