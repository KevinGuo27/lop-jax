from typing import Literal

from jax import numpy as jnp
from tap import Tap

class Hyperparams(Tap):
    study_name: str = 'test'
    seed: int = 2024
    debug: bool = False
    platform: Literal['cpu', 'gpu'] = 'gpu'
    n_seeds: int = 1

class ImagenetHyperparams(Hyperparams):
    env: str = 'imagenet'
    agent: Literal['er', 'bp', 'l2', 'snp_l2', 'snp', 'cbp', 'l2_er'] = 'l2_er'
    alg: Literal['actor_critic', 'ppo'] = 'ppo'
    activation: Literal['relu', 'tanh'] = 'relu'
    lr: list[float] = [0.01]
    optimizer: Literal['adam', 'sgd'] = 'sgd'
    weight_decay: float = 0.001 # Do we use L2 regularization?
    to_perturb: bool = False  # Whether to perturb the input data
    perturb_scale: float = 1e-5
    mini_batch_size: int = 100
    no_anneal_lr: bool = True
    max_grad_norm: float = 1e9
    num_tasks: int = 2000
    num_epochs: int = 250
    momentum: float = 0.9
    # Effective Rank
    er_lr: list[float] = [0.01]
    er_batch: int = 1
    er_step: int = 1

    # SVM
    svm: bool = False

    # Evaluation
    evaluate: bool = True # Do we evaluate after each task?
    evaluate_previous: bool = False  # Do we evaluate on previous tasks?
    eval_size: int = 2000
    compute_hessian: bool = False
    compute_hessian_size: int = 2000  # Number of samples to use for computing the hessian
    compute_hessian_interval: int = 1

    # CBP
    cont_backprop: bool = False
    replacement_rate: float = 1e-6
    decay_rate: float = 0.99
    maturity_threshold: int = 100

    def process_args(self) -> None:
        self.lr = jnp.array(self.lr)
        self.er_lr = jnp.array(self.er_lr)