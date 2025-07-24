#!/usr/bin/env python3
"""
Sanity check for incremental class loading with CifarDataSet.
"""
import numpy as np
from torch.utils.data import DataLoader, Subset
from mlproj_manager.problems import CifarDataSet
import torch
import jax.numpy as jnp

# Assumes loader_to_arrays and stratified_split are defined in the same package

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
        # ensure numpy
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            y = y.cpu().numpy()
        xs.append(x)
        ys.append(y)
    return jnp.array(np.concatenate(xs, axis=0)), jnp.array(np.concatenate(ys, axis=0))


def main():
    # Parameters for the sanity check
    classes_per_task = 5
    num_tasks = 3
    batch_size = 16
    val_fraction = 0.2

    # Initialize full CIFAR-100 dataset
    train_dataset = CifarDataSet(
        root_dir='./data/cifar-100-python',
        train=True,
        cifar_type=100,
        classes=np.array(range(1)),
        image_normalization=None,
        label_preprocessing=None,
        use_torch=False,
        flatten=False
    )

    test_dataset = CifarDataSet(
        root_dir='./data/cifar-100-python',
        train=False,
        cifar_type=100,
        classes=np.array(range(1)),
        image_normalization=None,
        label_preprocessing=None,
        use_torch=False,
        flatten=False
    )

    train_data = train_dataset.data['data']
    train_labels = train_dataset.data['labels']
    print(train_data.shape) # 
    print(train_labels.shape) 

    classes = [0]
    x_train = []
    for _class in classes:
        x_train.append(train_data[train_labels == _class])
    x_train = jnp.array(np.concatenate(x_train), dtype=jnp.float32).transpose(0, 2, 3, 1)

    print(x_train)

    print("Sanity check completed successfully.")


if __name__ == '__main__':
    main()