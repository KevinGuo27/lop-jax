from typing import NamedTuple

import chex
from flax.training.train_state import TrainState
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


def calculate_gae(traj_batch, last_val, last_done, gae_lambda, gamma: float):
    def _get_advantages(carry, transition):
        gae, next_value, next_done, gae_lambda = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, done, gae_lambda), gae

    _, advantages = jax.lax.scan(_get_advantages,
                                 (jnp.zeros_like(last_val), last_val, last_done, gae_lambda),
                                 traj_batch, reverse=True, unroll=16)
    target = advantages + traj_batch.value
    return advantages, target


class ActorCriticAgent:
    def __init__(self, network, args: PolicyHyperparams):
        self.value_loss_weight = args.value_loss_weight
        self.l2_reg_coeff = args.l2_reg_coeff
        self.network = network

    def act(self, rng: chex.PRNGKey, params: dict, obs: jnp.ndarray):
        obs = obs[None, :]
        pi, value = self.network.apply(params, obs)
        action = pi.sample(seed=rng)

        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        return value, action, log_prob

    def loss(self, params, traj_batch, returns):
        pi, value = self.network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_losses = jnp.square(value - returns)
        value_loss = value_losses.mean()

        # CALCULATE ACTOR LOSS
        actor_loss = -jnp.mean(log_prob * returns)

        total_loss = self.value_loss_weight * value_loss + actor_loss

        total_loss += l2_regularization(params, alpha=self.l2_reg_coeff)

        return total_loss, {'actor_loss': actor_loss, 'value_loss': value_loss}


