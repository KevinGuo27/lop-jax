from typing import Optional, Tuple, Union
from functools import partial

from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from brax.base import System
from brax.envs.base import State
import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env
        if hasattr(env, '_unwrapped'):
            self._unwrapped = env._unwrapped
        else:
            self._unwrapped = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    discounted_episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_discounted_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnums=(0))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_discounted_episode_return = state.discounted_episode_returns + (self.gamma ** state.episode_lengths) * reward
        new_episode_length = state.episode_lengths + 1
        # TODO: add discounted_episode_returns here.
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
                                     + new_episode_return * done,
            returned_discounted_episode_returns=state.returned_discounted_episode_returns * (1 - done)
                                                + new_discounted_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                     + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_discounted_episode_returns"] = state.returned_discounted_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["reward"] = reward
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        self.max_steps_in_episode = 1000
        env = EpisodeWrapper(env, episode_length=self.max_steps_in_episode, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


@struct.dataclass
class SlipperyAntState:
    env_state: environment.EnvState
    step: chex.Array


class AutoResetTimeStepWrapper(AutoResetWrapper):
    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        state.info['timestep'] = 0
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        state.info['timestep'] += 1
        return state


class NonstationaryFrictionBraxWrapper(BraxGymnaxWrapper):
    
    def __init__(self, env_name, backend="positional",
                 change_every: int = int(1e5),
                 lower_friction_exp: float = -4,
                 upper_friction_exp: float = 4):
        self.change_every = change_every
        self.lower_friction_exp = lower_friction_exp
        self.upper_friction_exp = upper_friction_exp

        env = envs.get_environment(env_name=env_name, backend=backend)
        self.max_steps_in_episode = 1000
        env = EpisodeWrapper(env, episode_length=self.max_steps_in_episode, action_repeat=1)
        env = AutoResetTimeStepWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def env_fn(self, sys: System):
        env = self._env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: chex.PRNGKey, params=None):
        state = self._env.reset(rng)
        sys = self._env.unwrapped.sys

        state.info['sys_variation'] = {'geom_friction': jnp.array(sys.geom_friction)}
        return state.obs, state

    def step(self, rng: chex.PRNGKey, state: State, action: jnp.ndarray, params=None):
        sys = self._env.unwrapped.sys
        new_timestep = state.info['timestep'] + 1

        def sample_new_friction(rng, s, a):
            friction_rng, rng = jax.random.split(rng)
            new_friction_exp = jax.random.uniform(friction_rng,
                                                  minval=self.lower_friction_exp,
                                                  maxval=self.upper_friction_exp)
            new_friction = 10 ** new_friction_exp
            new_geom_friction = s.info['sys_variation']['geom_friction'].at[:, 0].set(new_friction)
            new_variation = {'geom_friction': new_geom_friction}

            new_sys = sys.replace(**new_variation)
            new_env = self.env_fn(new_sys)

            s_prime = new_env.step(s, a)

            reset_rng, rng = jax.random.split(rng)
            new_s = new_env.reset(reset_rng)
            # TODO: test this somehow. I THINK this should be correct in terms of the done.
            new_s = new_s.replace(done=s_prime.done, reward=s_prime.reward)
            new_s.info['sys_variation'] = new_variation

            return new_s

        def use_same_friction(rng, s, a):
            new_sys = sys.replace(**s.info['sys_variation'])
            new_env = self.env_fn(new_sys)

            new_s = new_env.step(s, a)
            new_s.info['sys_variation'] = s.info['sys_variation']
            return new_s

        reset = new_timestep % self.change_every == 0
        next_state = jax.lax.cond(
            reset,
            sample_new_friction,
            use_same_friction,
            rng, state, action
        )
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

