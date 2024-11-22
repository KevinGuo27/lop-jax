from functools import partial
import inspect
from pathlib import Path
from typing import Union

import chex
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from gymnax.environments.spaces import Box, Discrete
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from rlopt.agents import ActorCriticAgent, PPOAgent, Transition
from rlopt.cbp import ContinualBackpropTrainState
from rlopt.config import NonStationaryPolicyHyperparams
from rlopt.envs import load_nonstationary_env, load_env, is_continuous, nonstationary_to_stationary_mapping
from rlopt.file_system import get_results_path
from rlopt.models import ActorCritic


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(rng: chex.PRNGKey, args: NonStationaryPolicyHyperparams):
    num_updates = (
        args.total_steps // args.num_steps // args.num_envs
    )
    minibatch_size = (
        args.num_envs * args.num_steps // args.num_minibatches
    )
    num_epochs = (
        num_updates // (args.change_every // args.num_steps // args.num_envs)
    )

    rng, _rng = jax.random.split(rng)
    env, env_params = load_nonstationary_env(rng, args.env, gamma=args.gamma)

    action_space = env.action_space(env_params)
    if isinstance(action_space, Box):
        action_shape = action_space.shape
        assert len(action_shape) == 1, "Can't handle action dim > 1"
        action_dim = action_shape[0]
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n

    network = ActorCritic(is_continuous=is_continuous(action_space),
                          action_dim=action_dim,
                          h_dims=(args.hidden_size,) * (args.num_hidden_layers + 1))

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_x)

    if args.alg == 'ppo':
        agent = PPOAgent(network, args)
    else:
        agent = ActorCriticAgent(network, args)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs))
            / num_updates
        )
        return args.lr * frac

    tstate_class = TrainState
    if args.cont_backprop:
        tstate_class = ContinualBackpropTrainState

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    updates_filter = partial(filter_period_first_dim, n=args.updates_log_freq)

    def train(rng):
        if args.no_anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        train_state = tstate_class.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        initial_train_state = train_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        def _epoch_step(runner_state, unused):

            # TRAIN LOOP
            def _update_step(runner_state, unused):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, unused):
                    train_state, env_state, last_obs, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    value, action, log_prob, activations = agent.act(_rng, train_state.params, last_obs)

                    if args.cont_backprop:
                        rng, _rng = jax.random.split(rng)
                        train_state = train_state.update_and_reinit(_rng,
                                                                    activations,
                                                                    replacement_rate=args.replacement_rate,
                                                                    decay_rate=args.decay_rate,
                                                                    maturity_threshold=args.maturity_threshold)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, args.num_envs)
                    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                    transition = Transition(
                        done, action, value, reward, log_prob, last_obs, info
                    )
                    runner_state = (train_state, env_state, obsv, rng)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, args.num_steps
                )

                # CALCULATE ADVANTAGE
                train_state, env_state, last_obs, rng = runner_state
                _, last_val, _ = network.apply(train_state.params, last_obs)

                advantages, targets = agent.target(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(runner_state, batch_info):
                        train_state, rng = runner_state
                        traj_batch, advantages, targets = batch_info

                        grad_fn = jax.value_and_grad(agent.loss, has_aux=True)

                        # TODO: return activations here.
                        loss_info, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        total_loss, losses_and_activations = loss_info

                        train_state = train_state.apply_gradients(grads=grads)

                        loss_info = (total_loss, losses_and_activations[:-1])

                        return (train_state, rng), loss_info

                    train_state, traj_batch, advantages, targets, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    # Batching and Shuffling
                    batch_size = minibatch_size * args.num_minibatches
                    assert (
                        batch_size == args.num_steps * args.num_envs
                    ), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (traj_batch, advantages, targets)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    # Mini-batch Updates
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x, [args.num_minibatches, -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, _rng), total_loss = jax.lax.scan(
                        _update_minbatch, (train_state, _rng), minibatches
                    )
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                # Updating Training State and Metrics:
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, args.update_epochs
                )
                train_state = update_state[0]
                metric = traj_batch.info
                metric = jax.tree.map(steps_filter, metric)
                rng = update_state[-1]

                # Debugging mode
                if args.debug:
                    def callback(info):
                        return_values = info["returned_episode_returns"][info["returned_episode"]]
                        timesteps = info["timestep"][info["returned_episode"]] * args.num_envs
                        for t in range(len(timesteps)):
                            print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                    jax.debug.callback(callback, metric)

                runner_state = (train_state, env_state, last_obs, rng)

                return runner_state, metric

            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, num_updates // num_epochs
            )
            metric = jax.tree.map(updates_filter, metric)

            return runner_state, (metric, runner_state)

        rng, _rng = jax.random.split(rng)
        init_runner_state = (train_state, env_state, obsv, _rng)
        final_runner_state, (metric, all_runner_states) = jax.lax.scan(
            _epoch_step, init_runner_state, None, num_epochs
        )

        final_train_state = final_runner_state[0]
        runner_states = {
            'initial_runner_state': init_runner_state,
            'intermediate_runner_states': all_runner_states,
            'final_runner_state': final_runner_state
        }

        return initial_train_state, final_train_state, metric, runner_states

    return train


def run_train(passed_in_args: Union[dict, NonStationaryPolicyHyperparams] = None) -> Path:
    # jax.disable_jit(True)
    ph = NonStationaryPolicyHyperparams()
    if isinstance(passed_in_args, NonStationaryPolicyHyperparams):
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

    init_train_states, final_train_states, metrics, runner_states = vmapped_train_fn(train_rngs)

    # remove methods from args
    dict_args = args.as_dict()
    for k in list(dict_args.keys()):
        if inspect.ismethod(dict_args[k]):
            del dict_args[k]

    all_results = {
        'init_train_state': init_train_states,
        'final_train_state': final_train_states,
        'runner_states': runner_states,
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
