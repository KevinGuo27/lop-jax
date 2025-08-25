from typing import Literal

from jax import numpy as jnp
from tap import Tap


class Hyperparams(Tap):
    study_name: str = 'test'
    seed: int = 2024
    debug: bool = False
    show_discounted: bool = False
    platform: Literal['cpu', 'gpu'] = 'cpu'

    def id_str(self):
        raise NotImplementedError

class PolicyHyperparams(Tap):
    env: str = 'slippery_ant'
    num_envs: int = 1
    gamma: float = 0.99

    num_steps: int = 1024
    update_epochs: int = 10
    num_minibatches: int = 16
    activation: Literal['relu', 'tanh'] = 'relu'
    optimizer: Literal['adam', 'sgd'] = 'adam'

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]
    vf_coeff: list[float] = [0.5]
    weight_decay: float = 0.0
    beta_1: float = 0.9
    beta_2: float = 0.999

    # Continual Backprop
    cont_backprop: bool = False
    replacement_rate: float = 1e-4
    decay_rate: float = 0.99
    maturity_threshold: int = int(1e4)

    # Effective Rank
    er: bool = False
    er_lr: list[float] = [0.01]
    er_batch: int = 1
    er_step: int = 1

    # compute hessian
    compute_hessian_init: bool = False
    compute_hessian_end: bool = False
    compute_hessian_size: int = 2000
    compute_hessian_interval: int = 1

    hidden_size: int = 256
    total_steps: int = int(5e6)
    entropy_coeff: float = 0.01
    clip_eps: float = 0.2
    max_grad_norm: float = 1e9
    anneal_lr: bool = False

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_checkpoints: bool = False  # Do we save train_state along with our per timestep outputs?
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 3
    platform: Literal['cpu', 'gpu'] = 'gpu'
    debug: bool = False
    show_discounted: bool = False  # For debug plotting, do we show undisc returns or disc returns?

    study_name: str = 'batch_ppo_test'

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
        self.er_lr = jnp.array(self.er_lr)


class PolicyEvalHyperparams(Hyperparams):
    checkpoint_path: str
    n_episodes: int = 1000
    n_bins: int = 100
    episodes_per_batch: int = 50
    max_episode_steps: int = 500

    # CARTPOLE
    cartpole_gravity_offset: float = 0.


class NonStationaryPolicyHyperparams(PolicyHyperparams):
    change_every: int = int(1e6)
    friction_seed: int = 0  # Seed for the friction schedule
