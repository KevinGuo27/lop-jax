from pathlib import Path

import chex
from flax.training.train_state import TrainState
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from rlopt.envs import LogWrapper, VecEnv
from rlopt.actor_critic import ActorCriticAgent, Transition
from rlopt.config import PolicyEvalHyperparams, PolicyHyperparams
from rlopt.models import Actor, ActorCritic


def first_nonzero_element(arr, axis=-1):
    """
    finds the first non-zero element along an axis.
    Defaults to -1.
    """
    indices = (arr != 0).argmax(axis=axis)

    return jnp.squeeze(jnp.take_along_axis(arr, jnp.expand_dims(indices, axis), axis=axis), axis)


def load_train_state(fpath: Path):
    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    args = PolicyHyperparams().from_dict(args)

    env, env_params = gymnax.make(args.env)
    env = LogWrapper(env, gamma=args.gamma)

    # Vectorize our environment
    env = VecEnv(env)

    network = ActorCritic(env.action_space(env_params), hidden_size=args.hidden_size)
    agent = ActorCriticAgent(network, args)

    ts_dict = restored['final_train_state']

    return env, env_params, args, agent, ts_dict['params'], {'metric': restored['metric']}


def make_policy_eval(args: PolicyEvalHyperparams):
    n_batches = args.n_episodes // args.episodes_per_batch
    env, env_params, train_args, agent, batch_train_state, info = load_train_state(args.checkpoint_path)

    def _env_step(runner_state, unused):
        network_params, env_state, last_obs, last_done, rng = runner_state
        rng, _rng = jax.random.split(rng)
        value, action, log_prob = agent.act(_rng, network_params, last_obs)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, args.episodes_per_batch)
        obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
        transition = (info, reward, done)
        runner_state = (network_params, env_state, obsv, done, rng)
        return runner_state, transition

    def policy_eval(env_state, obs: jnp.ndarray, rng: chex.PRNGKey, pi_params: dict):
        """
        Policy eval evaluates a policy (params) from the starting env_state.
        """
        def _run_episode(runner_state, i):
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.max_episode_steps), args.max_episode_steps
            )

            if args.debug:
                jax.debug.print("Running episode {}", i)
            return runner_state, traj_batch

        runner_state = (
            pi_params,
            jax.tree.map(lambda x: jnp.expand_dims(x, axis=0).repeat(args.episodes_per_batch, axis=0), env_state),
            obs[None, :].repeat(args.episodes_per_batch, axis=0),
            jnp.zeros((args.episodes_per_batch), dtype=bool),
            rng,
        )

        runner_state, traj_batch = jax.lax.scan(
            _run_episode, runner_state, jnp.arange(n_batches), n_batches
        )
        steps_last_traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, -1, -2), traj_batch)
        return steps_last_traj_batch

    return policy_eval, batch_train_state, train_args, env, env_params, info


def run_and_save_pe(passed_in_args: dict = None):
    # jax.disable_jit(True)
    peh = PolicyEvalHyperparams()
    if passed_in_args is not None:
        args = peh.from_dict(passed_in_args)
    else:
        args = peh.parse_args()

    jax.config.update('jax_platform_name', args.platform)

    res_dir = Path(args.checkpoint_path)
    new_results_dir = res_dir.parent / (res_dir.name + '_policy_eval_results')
    new_results_dir.mkdir(exist_ok=True)
    res_path = new_results_dir / 'parsed_results.npy'

    rng = jax.random.PRNGKey(args.seed)

    policy_eval_fn, pi_params, train_args, env, env_params, info = make_policy_eval(args)

    pi_params_1 = jax.tree.map(lambda x: x[0], pi_params)
    pi_params_2 = jax.tree.map(lambda x: x[1], pi_params)

    taus = jnp.linspace(0, 1, num=args.n_bins)

    def interpolate_params(x1, x2):
        taus_shape = tuple(range(1, len(x1.shape) + 1))
        expanded_taus = jnp.expand_dims(taus, axis=taus_shape)
        return expanded_taus * x1[None, ...] + (1 - expanded_taus) * x2[None, ...]

    interpolated_pi_params = jax.tree.map(interpolate_params, pi_params_1, pi_params_2)

    vmapped_policy_eval_fn = jax.vmap(policy_eval_fn, in_axes=[None, None, 0, 0])

    rng, reset_rng = jax.random.split(rng)
    reset_rng = reset_rng[None, :]
    obsv, env_state = env.reset(reset_rng, env_params)

    obsv = obsv[0]
    env_state = jax.tree.map(lambda x: x[0], env_state)

    rng, pe_rng = jax.random.split(rng)
    pe_rngs = jax.random.split(pe_rng, args.n_bins)

    traj = vmapped_policy_eval_fn(env_state, obsv, pe_rngs, interpolated_pi_params)

    disc_returns = first_nonzero_element(traj[0]['returned_discounted_episode_returns']).reshape(args.n_bins, -1)
    returns = first_nonzero_element(traj[0]['returned_episode_returns']).reshape(args.n_bins, -1)

    train_disc_rets = info['metric']['returned_discounted_episode_returns']
    train_disc_rets = train_disc_rets.reshape(train_disc_rets.shape[0], -1, train_disc_rets.shape[-1])  # n_params x total_steps x num_envs
    train_rets = info['metric']['returned_episode_returns']
    train_rets = train_rets.reshape(train_rets.shape[0], -1, train_rets.shape[-1])

    dict_args = args.as_dict()
    del dict_args['id_str']

    res = {
        'train_args': train_args,
        'eval_args': dict_args,
        'taus': taus,
        'interpolated_discounted_returns': disc_returns,
        'interpolated_returns': returns,
        'training_discounted_returns': train_disc_rets,
        'training_returns': train_rets
    }

    res = jax.tree.map(lambda x: np.array(x), res)

    print(f'Saving to {res_path}')

    np.save(res_path, res)
    print('Done')

    return res_path


if __name__ == "__main__":
    run_and_save_pe()
