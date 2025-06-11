from typing import Literal

from jax import numpy as jnp
from tap import Tap

class Hyperparams(Tap):
    study_name: str = 'test'
    seed: int = 2024
    debug: bool = False
    platform: Literal['cpu', 'gpu'] = 'gpu'
    n_seeds: int = 1

class PermutedMnistHyperparams(Hyperparams):
    env: str = 'permuted_mnist'
    agent: Literal['er', 'bp', 'l2', 'cbp', 'l2_er'] = 'l2_er'
    alg: Literal['actor_critic', 'ppo'] = 'ppo'
    activation: Literal['relu', 'tanh'] = 'relu'
    lr: list[float] = [0.01]
    optimizer: Literal['adam', 'sgd'] = 'sgd'
    weight_decay: float = 0.001 # Do we use L2 regularization?
    num_features: int = 100  # Number of input features
    change_after: int = 10 * 6000  # Number of steps after which the task changes
    to_perturb: bool = False  # Whether to perturb the input data
    perturb_scale: int = 0.1
    num_hidden_layers: int = 3
    mini_batch_size: int = 1
    no_anneal_lr: bool = True
    max_grad_norm: float = 0.5
    num_tasks: int = 800  # Number of tasks in the permuted MNIST

    # Effective Rank
    er_lr: list[float] = [0.01]
    er_batch: int = 100
    er_step: int = 1

    # Evaluation
    evaluate: bool = True # Do we evaluate after each task?
    eval_size: int = 2000
    record_hessian: bool = False
    record_relu: bool = False
    compute_hessian: bool = False

    def process_args(self) -> None:
        self.lr = jnp.array(self.lr)
        self.er_lr = jnp.array(self.er_lr)