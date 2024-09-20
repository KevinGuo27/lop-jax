from typing import Literal

from jax import numpy as jnp
from tap import Tap


class PolicyHyperparams(Tap):
    env: str = 'CartPole-v1'
    alg: Literal['actor_critic'] = 'actor_critic'
    lr: float = 3e-4
    value_loss_weight: float = 0.
    hidden_size: int = 32

    gamma: float = 0.95
    num_steps: int = 512  # How many steps in our n-step returns?

    total_steps: int = int(1e6)
    num_envs: int = 4
    n_param_sets: int = 2

    seed: int = 2024
    debug: bool = False
    platform: Literal['cpu', 'gpu'] = 'cpu'
