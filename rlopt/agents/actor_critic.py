from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from rlopt.config import PolicyHyperparams
from rlopt.utils import l2_regularization


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def compute_n_step_returns(traj_batch: Transition, last_vals, gamma: float):

    def _step(prev_vals, runner):
        terminal, reward = runner
        vals = reward + gamma * (1 - terminal) * prev_vals
        return vals, vals

    _, returns = jax.lax.scan(
        _step, last_vals,
        (traj_batch.done, traj_batch.reward),
        reverse=True
    )
    return returns


class ActorCriticAgent:
    def __init__(self, network, args: PolicyHyperparams):
        self.value_loss_weight = args.value_loss_weight
        self.l2_reg_coeff = args.l2_reg_coeff
        self.gamma = args.gamma
        self.network = network

    def act(self, rng: chex.PRNGKey, params: dict, obs: jnp.ndarray):
        pi, value = self.network.apply(params, obs)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return value, action, log_prob

    def loss(self, params, traj_batch, returns, value_targets, return_intermediates: bool = False):
        intermediates = None
        if not return_intermediates:
            pi, value = self.network.apply(params, traj_batch.obs)
        else:
            return_tuple, intermediates = self.network.apply(params, traj_batch.obs, capture_intermediates=True,
                                                             mutable=['intermediates'])
            pi = return_tuple[0]
            value = return_tuple[1]
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_losses = jnp.square(value - value_targets)
        value_loss = value_losses.mean()

        # CALCULATE ACTOR LOSS
        actor_loss = -jnp.mean(log_prob * returns)

        total_loss = self.value_loss_weight * value_loss + actor_loss

        total_loss += l2_regularization(params, alpha=self.l2_reg_coeff)
        if not return_intermediates:
            return total_loss, {'actor_loss': actor_loss, 'value_loss': value_loss}
        else:
            return total_loss, {'actor_loss': actor_loss, 'value_loss': value_loss, 'intermediates': intermediates}

    def target(self, traj_batch: Transition, last_vals: chex.Array):
        # N step returns
        def _step(prev_vals, runner):
            terminal, reward = runner
            vals = reward + self.gamma * (1 - terminal) * prev_vals
            return vals, vals

        _, returns = jax.lax.scan(
            _step, last_vals,
            (traj_batch.done, traj_batch.reward),
            reverse=True
        )
        return returns, returns


