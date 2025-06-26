import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from jax_bp import create_train_state, train_step, compute_accuracy
from jax_convnet import ConvNet
import jax
import jax.numpy as jnp

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
    opt = params['opt']
    num_classes = params['num_classes']
    mini_batch_size = 100
    momentum = 0
    if 'momentum' in params.keys(): momentum = params['momentum']
    num_epochs = num_showings

    classes_per_task = num_classes

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones([1, 32, 32, 3])
    state = create_train_state(rng, step_size, momentum, ConvNet, dummy_input)

    with open('class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[int([params['run_idx']][0])]
    num_class_repetitions_required = int(num_classes * num_tasks / 1000) + 1
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

        for epoch_idx in tqdm(range(num_epochs)):
            example_order = np.random.permutation(train_images_per_class * classes_per_task)
            new_train_accuracies = []
            for start_idx in range(0, examples_per_epoch, mini_batch_size):
                
                batch_indices = example_order[start_idx: start_idx+mini_batch_size]
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                batch = {'image': batch_x, 'label': batch_y}

                state, loss, logits = train_step(state, batch)
                new_train_accuracies.append(compute_accuracy(logits, batch_y))

            train_accuracies[task_idx][epoch_idx] = np.mean(new_train_accuracies)

            new_test_accuracies = []
            for start_idx in range(0, x_test.shape[0], mini_batch_size):
                test_batch_x = x_test[start_idx: start_idx + mini_batch_size]
                test_batch_y = y_test[start_idx: start_idx + mini_batch_size]
                logits = state.apply_fn({'params': state.params}, test_batch_x)
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
    params = {
        'agent': 'bp',
        'num_tasks': 10,
        'num_showings': 10,
        'step_size': 0.001,
        'opt': 'sgd',
        'run_idx': 0,
        'num_classes': 2,
        'data_file': 'data/imagenet_bp_jax.pkl',
    }
    repeat_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
