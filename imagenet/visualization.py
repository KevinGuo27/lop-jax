import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import jax.numpy as jnp
from pathlib import Path

train_images_per_class = 600
test_images_per_class = 100

def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        print(f"Loading class {idx} of {len(classes)}")
        data_file = '/users/kguo32/rl-opt/imagenet/data/classes/' + str(_class) + '.npy'
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

def plot_task_images(x, y, classes=None, samples_per_class=5,
                     class_names=None, figsize=(10, 10), save_path=None):
    """Plot grid of sample images per class."""
    x = np.array(x)
    y = np.array(y)
    if classes is None:
        classes = np.unique(y)
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=figsize, squeeze=False)
    for i, cls in enumerate(classes):
        idxs = np.where(y == cls)[0]
        selected = np.random.choice(idxs,
                                    min(len(idxs), samples_per_class),
                                    replace=False)
        for j, idx in enumerate(selected):
            ax = axes[i, j]
            img = x[idx]
            y_idx = y[idx]
            print(f"Class {y_idx}")
            if img.dtype != np.uint8:
                img = (img - img.min()) / (max(img.max() - img.min(), 1e-8))
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                label = class_names.get(cls, str(cls)) if class_names else str(cls)
                ax.set_ylabel(label, rotation=0, labelpad=40, va='center')
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    with open('/users/kguo32/rl-opt/imagenet/class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[0]
    
    parser = argparse.ArgumentParser(description='Visualize ImageNet task samples')
    parser.add_argument('--task', type=int, required=True,
                        help='Task index to visualize')
    parser.add_argument('--output_file', type=str, default='visualization.png')
    args = parser.parse_args()
    classes_per_task = 2
    x_train, y_train, x_test, y_test = load_imagenet(class_order[args.task*classes_per_task:(args.task+1)*classes_per_task])
    plot_task_images(x_train, y_train,
                     save_path=args.output_file)