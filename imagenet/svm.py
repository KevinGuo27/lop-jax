import torch
import pickle
import argparse
import numpy as np
from sklearn import svm

def load_imagenet(classes=[]):
    train_images_per_class = 600
    test_images_per_class = 100
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = '/users/kguo32/rl-opt/imagenet/data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    # flatten the images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_test, y_test

def main(args):
    classes_per_task = 2
    with open('/users/kguo32/rl-opt/imagenet/class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[0]
    for task in range(args.task_init, args.task_end):
        x_train, y_train, x_eval, y_eval = load_imagenet(class_order[task*classes_per_task:(task+1)*classes_per_task])
        model = svm.SVC(kernel='linear', probability=True)
        model.fit(x_train, y_train)
        # evaluate the model
        acc = model.score(x_eval, y_eval)
        print(f'Task {task}, Accuracy: {acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incremental SVM over subsets of ImageNet classes."
    )
    parser.add_argument(
        "--task_init",
        type=int,
        default=0,
        help="Index of the first task to run (0-based)."
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=0,
        help="Index of the last task to run (0-based)."
    )
    args = parser.parse_args()
    main(args)