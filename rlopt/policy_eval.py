from functools import partial
from typing import Union

import chex
import jax
import jax.numpy as jnp

from rlopt.agents.actor_critic import ActorCriticAgent, Transition
from rlopt.agents.ppo import PPOAgent


def env_step(runner_state, unused, agent, episodes, env, env_params):
    network_params, env_state, last_obs, last_done, rng = runner_state
    rng, _rng = jax.random.split(rng)
    value, action, log_prob, _ = agent.act(_rng, network_params, last_obs)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, episodes)
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    transition = Transition(
        done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (network_params, env_state, obsv, done, rng)
    return runner_state, transition


def policy_eval(env_state, obs: jnp.ndarray, rng: chex.PRNGKey, pi_params: dict,
                env, env_params,
                agent: Union[PPOAgent, ActorCriticAgent],
                max_episode_steps: int = 1000,
                n_episodes: int = 1000,
                episodes_per_batch: int = 50,
                debug: bool = True):
    """
    Policy eval evaluates a policy (params) from the starting env_state.
    """
    n_batches = n_episodes // episodes_per_batch

    _env_step = partial(env_step, agent=agent, episodes=episodes_per_batch, env=env, env_params=env_params)

    def _run_episode(runner_state, i):
        new_rs, traj_batch = jax.lax.scan(
            _env_step, runner_state, jnp.arange(max_episode_steps), max_episode_steps
        )

        _, _, last_obs, last_done, rng = new_rs
        _, last_val, _ = agent.network.apply(pi_params, last_obs)

        xtra = {}
        if isinstance(agent, PPOAgent):
            # TODO: currently this is done with values predicted from the
            # interpolated value func. We might want to do it from either ends of the interpolation
            advantages, gae = agent.target(traj_batch, last_val)
            xtra['advantages'] = advantages
            xtra['gae'] = gae

        if debug:
            jax.debug.print("Running episode {}", i)

        # same runner state going thru, except for rng
        next_runner_state = runner_state[:-1] + (rng,)
        return next_runner_state, (traj_batch, xtra)

    runner_state = (
        pi_params,
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0).repeat(episodes_per_batch, axis=0), env_state),
        obs[None, :].repeat(episodes_per_batch, axis=0),
        jnp.zeros((episodes_per_batch), dtype=bool),
        rng,
    )
    runner_state, (traj_batch, xtra_info) = jax.lax.scan(
        _run_episode, runner_state, jnp.arange(n_batches), n_batches
    )
    steps_last_traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, -1, -2), traj_batch)
    steps_last_xtra_info = jax.tree.map(lambda x: jnp.swapaxes(x, -1, -2), xtra_info)

    return steps_last_traj_batch, steps_last_xtra_info

