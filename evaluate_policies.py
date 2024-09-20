from pathlib import Path

import gymnax
import jax
import orbax.checkpoint
from rlopt.envs import LogWrapper, VecEnv

from rlopt.config import PolicyEvalHyperparams


def load_train_state(key: jax.random.PRNGKey, fpath: Path):
    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    unpacked_ts = restored['out']['runner_state'][0]

    env, env_params = gymnax.make(args.env)
    env = LogWrapper(env, gamma=args.gamma)

    network_fn, action_size = get_network_fn(env, env_params, memoryless=args['memoryless'])

    network = network_fn(action_size,
                         double_critic=args['double_critic'],
                         hidden_size=args['hidden_size'])
    tx = optax.adam(args['lr'][0])
    ts = TrainState.create(apply_fn=network.apply,
                           params=jax.tree_map(lambda x: x[0, 0, 0, 0, 0, 0], unpacked_ts['params']),
                           tx=tx)

    return env, env_params, args, network, ts


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = PolicyEvalHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)

    env, env_params, args, network, ts = load_train_state(rng, args.checkpoint_path)
