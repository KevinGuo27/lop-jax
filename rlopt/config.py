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
    alg: Literal['actor_critic', 'ppo'] = 'actor_critic'
    lr: float = 2.5e-4
    hidden_size: int = 32
    num_hidden_layers: int = 1
    l2_reg_coeff: float = 0.  # Do we use L2 regularization?
    num_minibatches: int = 4

    # Actor Critic
    value_loss_weight: float = 0.

    gamma: float = 0.95
    num_steps: int = 128  # How many steps in our n-step returns?

    # Continual Backprop
    cont_backprop: bool = False
    replacement_rate: float = 1e-4
    decay_rate: float = 0.99
    maturity_threshold: int = int(1e4)

    # PPO
    entropy_coeff: float = 0.01
    vf_coeff: float = 0.5
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    adv_lambda: float = 0.95
    update_epochs: int = 4

    total_steps: int = int(1e7)
    num_envs: int = 4
    n_param_sets: int = 2
    steps_log_freq: int = 128
    updates_log_freq: int = 100

    def id_str(self):
        return f"{self.env}_{self.alg}_seed({self.seed})"


class PolicyEvalHyperparams(Hyperparams):
    checkpoint_path: str
    n_episodes: int = 1000
    n_bins: int = 100
    episodes_per_batch: int = 50
    max_episode_steps: int = 500

    # CARTPOLE
    cartpole_gravity_offset: float = 0.
