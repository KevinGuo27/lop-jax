"""
Trains a set of parameters with n different seeds.
"""
import inspect
from functools import partial

import chex
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from rlopt.actor_critic import ActorCriticAgent, Transition, compute_n_step_returns
from rlopt.config import PolicyHyperparams
from rlopt.envs import LogWrapper, VecEnv
from rlopt.models import Actor, ActorCritic
from rlopt.file_system import get_results_path


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(rng: chex.PRNGKey):
    """
    Make the training function. Namely, initialize the parameters we'll
    be optimizing.
    """
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )

    env, env_params = gymnax.make(args.env)
    env = LogWrapper(env, gamma=args.gamma)

    # Vectorize our environment
    env = VecEnv(env)

    # TODO: refactor this to allow continuous actions
    # network = Actor(env.action_space(env_params).n, hidden_size=args.hidden_size)
    network = ActorCritic(env.action_space(env_params).n, hidden_size=args.hidden_size)

    agent = ActorCriticAgent(network, args)

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, args.num_envs, *env.observation_space(env_params).shape))

    network_params = network.init(_rng, init_x)

    # TODO: what optimizer do we use?
    tx = optax.adam(learning_rate=args.lr)

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
            value, action, log_prob = agent.act(_rng, train_state, last_obs)

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

            train_state, env_state, obsv, done, rng = runner_state
            _, last_val = network.apply(train_state.params, obsv)

            # CALCULATE RETURNS
            vmapped_compute_n_step_returns = jax.vmap(compute_n_step_returns, in_axes=[1, 0, None], out_axes=1)
            returns = vmapped_compute_n_step_returns(traj_batch, last_val, args.gamma)
            batch = (returns, traj_batch)

            # flatten everything
            def flatten_first_n(arr: jnp.ndarray, n: int):
                return arr.reshape(-1, *arr.shape[n:])
            flatten_first_two = partial(flatten_first_n, n=2)

            flat_batch = jax.tree.map(flatten_first_two, batch)
            flat_returns, flat_traj = flat_batch

            # now we need to shuffle everything
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, flat_returns.shape[0])

            shuffled_returns, shuffled_traj = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), flat_batch
            )

            # Now update our params
            grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, shuffled_traj, shuffled_returns
            )
            train_state = train_state.apply_gradients(grads=grads)
            runner_state = (train_state, env_state, obsv, done, rng)

            # save metrics only every steps_log_freq
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)

            if args.debug:

                def callback(info):
                    timesteps = (
                            info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])
                    if len(timesteps) > 0:
                        # jax.debug.print(
                        #     "timesteps={} - {}, avg episodic return={:.2f}, actor_loss: {}, critic_loss: {}",
                        #     timesteps[0], timesteps[-1], avg_return_values,
                        #     total_loss[1]['actor_loss'], total_loss[1]['value_loss']
                        # )
                        jax.debug.print(
                            "timesteps={} - {}, avg episodic return={:.2f}",
                            timesteps[0], timesteps[-1], avg_return_values
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


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = PolicyHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    rng, make_train_rng = jax.random.split(rng)

    train_fn = make_train(make_train_rng)
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

