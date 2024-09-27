from typing import Literal

from jax import numpy as jnp
from tap import Tap


class Hyperparams(Tap):
    study_name: str = 'test'
    seed: int = 2024
    debug: bool = False
    platform: Literal['cpu', 'gpu'] = 'cpu'

    def id_str(self):
        raise NotImplementedError


class PolicyHyperparams(Hyperparams):
    env: str = 'CartPole-v1'
    alg: Literal['actor_critic'] = 'actor_critic'
    lr: float = 1e-4
    value_loss_weight: float = 0.
    hidden_size: int = 32
    l2_reg_coeff: float = 0.  # Do we use L2 regularization?

    gamma: float = 0.95
    num_steps: int = 1000  # How many steps in our n-step returns?

    total_steps: int = int(5e7)
    num_envs: int = 4
    n_param_sets: int = 2
    steps_log_freq: int = 128
    updates_log_freq: int = 100

    def id_str(self):
        return f"{self.env}_seed({self.seed})"


class PolicyEvalHyperparams(Hyperparams):
    checkpoint_path: str
    n_episodes: int = 1000
    n_bins: int = 100
    episodes_per_batch: int = 50
    max_episode_steps: int = 500

    # CARTPOLE
    cartpole_gravity_offset: float = 0.
