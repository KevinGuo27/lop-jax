"""
Trains a set of parameters with n different seeds.
"""
from functools import partial
import inspect
from pathlib import Path
from typing import Union

import chex
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from rlopt.agents import ActorCriticAgent, PPOAgent, Transition
from rlopt.envs import load_env
from rlopt.config import PolicyHyperparams
from rlopt.models import ActorCritic
from rlopt.file_system import get_results_path


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(rng: chex.PRNGKey, args: PolicyHyperparams):
    """
    Make the training function. Namely, initialize the parameters we'll
    be optimizing.
    """
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )

    env, env_params = load_env(args.env, gamma=args.gamma)

    network = ActorCritic(env.action_space(env_params), hidden_size=args.hidden_size)

    if args.alg == 'ppo':
        agent = PPOAgent(network, args)
    else:
        agent = ActorCriticAgent(network, args)

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (args.num_envs * args.update_epochs))
                / num_updates
        )
        return args.lr * frac

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, args.num_envs, *env.observation_space(env_params).shape))

    network_params = network.init(_rng, init_x)

    # TODO: what optimizer do we use?
    if args.alg == 'ppo' and args.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-5),
        )

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    updates_filter = partial(filter_period_first_dim, n=args.updates_log_freq)

    def train(rng):
        init_train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, rng = runner_state
            rng, _rng = jax.random.split(rng)
            value, action, log_prob = agent.act(_rng, train_state.params, last_obs)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, args.num_envs)
            obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info
            )
            runner_state = (train_state, env_state, obsv, done, rng)
            return runner_state, transition

        # TRAIN LOOP
        def _update_step(runner_state, i):
            # COLLECT TRAJECTORIES
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            train_state, env_state, final_obs, final_done, rng = runner_state
            _, final_val = network.apply(train_state.params, final_obs)

            # CALCULATE TARGETS
            vmapped_target_fn = jax.vmap(agent.target, in_axes=[1, 0, 0], out_axes=1)
            returns, value_targets = vmapped_target_fn(traj_batch, final_val, final_done)
            batch = (returns, value_targets, traj_batch)

            # flatten everything
            def flatten_first_n(arr: jnp.ndarray, n: int):
                return arr.reshape(-1, *arr.shape[n:])
            flatten_first_two = partial(flatten_first_n, n=2)

            flat_batch = jax.tree.map(flatten_first_two, batch)
            flat_returns, flat_value_targets, flat_traj = flat_batch

            # now we need to shuffle everything
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, flat_value_targets.shape[0])

            shuffled_returns, shuffled_value_targets, shuffled_traj = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), flat_batch
            )

            # Now update our params
            grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, shuffled_traj, shuffled_returns, shuffled_value_targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            runner_state = (train_state, env_state, final_obs, final_done, rng)

            # save metrics only every steps_log_freq
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)

            if args.debug:

                def callback(info):
                    avg_return_values = jnp.mean(info["returned_episode_returns"])
                    jax.debug.print(
                        "timesteps={} - {}, avg episodic return={:.2f}",
                        info['timestep'].min(), info['timestep'].max(), avg_return_values
                    )

                jax.debug.callback(callback, metric)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            init_train_state,
            env_state,
            obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            _rng,
        )

        # returned metric has an extra dimension.
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates), num_updates
        )

        final_train_state = runner_state[0]
        metric = jax.tree.map(updates_filter, metric)  # update_steps x (args.num_steps // args.steps_log_freq) x num_envs

        return init_train_state, final_train_state, metric

    return train


def run_train(passed_in_args: Union[dict, PolicyHyperparams] = None) -> Path:
    # jax.disable_jit(True)
    ph = PolicyHyperparams()
    if isinstance(passed_in_args, PolicyHyperparams):
        args = passed_in_args
    elif passed_in_args is not None:
        args = ph.from_dict(passed_in_args)
    else:
        args = ph.parse_args()

    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    rng, make_train_rng = jax.random.split(rng)

    train_fn = make_train(make_train_rng, args)
    jitted_train_fn = jax.jit(train_fn)

    # now we vmap rng over n_param_sets
    rng, train_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, args.n_param_sets)

    vmapped_train_fn = jax.vmap(jitted_train_fn)

    init_train_states, final_train_states, metrics = vmapped_train_fn(train_rngs)

    # remove methods from args
    dict_args = args.as_dict()
    for k in list(dict_args.keys()):
        if inspect.ismethod(dict_args[k]):
            del dict_args[k]

    all_results = {
        'init_train_state': init_train_states,
        'final_train_state': final_train_states,
        'metric': metrics,
        'args': dict_args
    }

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)

    print("Done.")
    return results_path


if __name__ == "__main__":
    run_train()
