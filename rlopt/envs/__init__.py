from functools import partial

from brax.envs import _envs as brax_envs
import chex
from gymnax.environments.spaces import Box, Discrete, Space
import gymnax
from gymnax import EnvParams
import jax

from .wrappers import BraxGymnaxWrapper, LogWrapper, ClipAction, VecEnv, SlipperyAntWrapper


def load_brax_env(env_str: str):
    env = BraxGymnaxWrapper(env_str)
    env_params = EnvParams(max_steps_in_episode=env.max_steps_in_episode)
    env = ClipAction(env)
    return env, env_params


def load_env(env_str: str, gamma: float):
    if env_str in brax_envs:
        env, env_params = load_brax_env(env_str)
    else:
        env, env_params = gymnax.make(env_str)

    env = LogWrapper(env, gamma=gamma)

    # Vectorize our environment
    env = VecEnv(env)
    return env, env_params


def load_nonstationary_env(rng: chex.PRNGKey, env_str: str, gamma: float,
                           change_every: int = int(1e6)):
    assert env_str in ['slippery_ant']
    if env_str == 'slippery_ant':
        env_str = 'ant'
        friction_rng, key = jax.random.split(rng)
        new_friction_exp = jax.random.uniform(friction_rng, minval=-4, maxval=4)
        new_friction = 10 ** new_friction_exp
        # TODO:
        env, env_params = load_brax_env(env_str)
    else:
        raise NotImplementedError

    env = LogWrapper(env, gamma=gamma)

    # Vectorize our environment
    env = VecEnv(env)
    return env, env_params


def is_continuous(space: Space):
    if isinstance(space, Box):
        return True
    elif isinstance(space, Discrete):
        return False
    else:
        raise NotImplementedError
