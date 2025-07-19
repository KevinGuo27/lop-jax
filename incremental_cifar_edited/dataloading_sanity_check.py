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
    dataset = CifarDataSet(
        root_dir='./data/cifar-100-python',
        train=True,
        cifar_type=100,
        classes=None,
        image_normalization=None,
        label_preprocessing=None,
        use_torch=False,
        flatten=False
    )

    for task in range(num_tasks):
        # Define the new classes for this task
        new_classes = list(range(task * classes_per_task,
                                 (task + 1) * classes_per_task))
        dataset.select_new_partition(new_classes)

        # Check that only the intended classes are present
        current_labels = np.array(dataset.integer_labels)[
            np.in1d(dataset.integer_labels, new_classes)
        ]
        unique_labels = np.unique(current_labels)
        print(f"Task {task}: expecting classes {new_classes}, found {list(unique_labels)}")

        # Stratified split into train/validation
        labels_array = np.array(current_labels)
        train_idx, val_idx = stratified_split(labels_array, val_fraction)

        # Build PyTorch DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size,
                                  shuffle=False)
        val_loader = DataLoader(Subset(dataset, val_idx),
                                batch_size=batch_size,
                                shuffle=False)

        # Convert loaders to JAX arrays
        x_train, y_train = loader_to_arrays(train_loader)
        x_val, y_val = loader_to_arrays(val_loader)
        x_test, y_test = loader_to_arrays(val_loader)
        x_all, y_all = jnp.concatenate([x_train, x_val, x_test], axis=0), jnp.concatenate([y_train, y_val, y_test], axis=0)

        # Print shapes to verify
        print(f"  Train: x={x_train.shape}, y={y_train.shape}")
        print(f"  Val:   x={x_val.shape}, y={y_val.shape}")
        print(f"  Test:  x={x_test.shape}, y={y_test.shape}")
        print(f" All:  x={x_all.shape}, y={y_all.shape}") # should be of shape (3000, 32, 32, 3)
        print("---")

    print("Sanity check completed successfully.")


if __name__ == '__main__':
    main()