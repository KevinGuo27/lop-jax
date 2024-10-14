import chex
import jax
import jax.numpy as jnp

from rlopt.config import PolicyHyperparams
from rlopt.utils import l2_regularization

from .actor_critic import ActorCriticAgent, Transition


class PPOAgent(ActorCriticAgent):
    def __init__(self, network,
                 args: PolicyHyperparams):
        super().__init__(network, args)
        self.adv_lambda = args.adv_lambda
        self.entropy_coeff = args.entropy_coeff
        self.clip_eps = args.clip_eps

    def act(self, rng: chex.PRNGKey,
            params: dict,
            obs: chex.Array):

        # SELECT ACTION
        ac_in = obs[None, :]
        pi, value = self.network.apply(params, ac_in)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)

        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        return value, action, log_prob

    def loss(self, params: dict, traj_batch: Transition, gae: jnp.ndarray, targets: jnp.ndarray):
        # RERUN NETWORK
        pi, value = self.network.apply(
            params, traj_batch.obs
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            jnp.maximum(value_losses, value_losses_clipped).mean()
        )
        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # which advantage do we use to update our policy?
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                    )
                * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
                loss_actor
                + value_loss
                - self.entropy_coeff * entropy
        )

        total_loss += l2_regularization(params, alpha=self.l2_reg_coeff)

        return total_loss, (value_loss, loss_actor, entropy)

    def target(self, traj_batch: Transition, last_vals: chex.Array, last_done):
        # Generalized Advantage Estimation

        def _get_advantages(carry, transition):
            gae, next_value, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + self.gamma * next_value * (1 - next_done) - value
            gae = delta + self.gamma * self.adv_lambda * (1 - next_done) * gae
            return (gae, value, done), gae

        _, advantages = jax.lax.scan(_get_advantages,
                                     (jnp.zeros_like(last_vals), last_vals, last_done),
                                     traj_batch, reverse=True, unroll=16)
        target = advantages + traj_batch.value
        return advantages, target
