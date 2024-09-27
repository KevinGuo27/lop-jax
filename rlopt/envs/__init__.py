from brax.envs import _envs as brax_envs
import gymnax
from gymnax import EnvParams

from .wrappers import BraxGymnaxWrapper, LogWrapper, ClipAction, VecEnv


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

