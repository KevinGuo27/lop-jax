from typing import NamedTuple

from collections import deque
from dataclasses import replace
from functools import partial
import inspect
from time import time

import chex
import flax.training.train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from gymnax.environments.spaces import Box, Discrete
from rlopt.envs.wrappers import LogEnvState
from rlopt.utils.file_system import get_results_path, numpyify
from rlopt.cbp import ContinualBackpropTrainState
from rlopt.config import NonStationaryPolicyHyperparams
from rlopt.envs import load_nonstationary_env, load_env, is_continuous
from rlopt.models import ActorCritic, ScannedRNN
from rlopt.utils.optimizer import adam_with_param_counts

from rlopt.utils.hessian_computation import get_hvp_fn
from rlopt.utils.lanczos import lanczos_alg
from rlopt.utils.file_system import plot_hessian_spectrum
from rlopt.utils.density import tridiag_to_density, tridiag_to_density_and_erank


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray = None

class PPO:
    def __init__(self, network,
                 vf_coeff: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2):
        self.network = network
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps
        self.act = jax.jit(self.act)
        self.loss = jax.jit(self.loss)

    def act(self, rng: chex.PRNGKey,
            train_state: flax.training.train_state.TrainState,
            hidden_state: chex.Array,
            obs: chex.Array, done: chex.Array):

        # SELECT ACTION
        ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
        hstate, pi, value, activations = self.network.apply(train_state.params, hidden_state, ac_in)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        return value, action, log_prob, hstate, activations

    def loss(self, params, init_hstate, traj_batch, gae, targets):
        # RERUN NETWORK
        _, pi, value, _ = self.network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
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
                + self.vf_coeff * value_loss
                - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


def env_step(runner_state, unused, agent: PPO, env, env_params, compute_hessian=False):
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)
    value, action, log_prob, hstate, activations = agent.act(_rng, train_state, hstate, last_obs, last_done)
    if args.cont_backprop and not compute_hessian:
        rng, _rng = jax.random.split(rng)
        train_state = train_state.update_and_reinit(_rng,
                                                    activations,
                                                    replacement_rate=args.replacement_rate,
                                                    decay_rate=args.decay_rate,
                                                    maturity_threshold=args.maturity_threshold)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    
    transition = Transition(
        last_done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, transition


def calculate_gae(traj_batch, last_val, last_done, gae_lambda, gamma):
    def _get_advantages(carry, transition):
        gae, next_value, gae_lambda = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value, gae_lambda), gae

    _, advantages = jax.lax.scan(_get_advantages,
                                 (jnp.zeros_like(last_val), last_val, gae_lambda),
                                 traj_batch, reverse=True, unroll=16)
    target = advantages + traj_batch.value
    return advantages, target


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(args: NonStationaryPolicyHyperparams, rand_key: jax.random.PRNGKey):
    num_updates = (
        args.total_steps // args.num_steps // args.num_envs
    )
    minibatch_size = (
        args.num_envs * args.num_steps // args.num_minibatches
    )
    num_epochs = (
        num_updates // (args.change_every // args.num_steps // args.num_envs)
    )
    num_friction_changes = (
        args.total_steps // args.change_every // args.num_envs
    )
    # randomly generate a friction schedule
    rng, _rng = jax.random.split(rand_key)
    friction_schedule = jax.random.uniform(_rng, (num_friction_changes,), minval=0, maxval=2)
    print('Friction schedule:', friction_schedule)
    env, env_params = load_nonstationary_env(rng, args.env, gamma=args.gamma, 
                        change_every=args.change_every, friction_schedule=friction_schedule)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    action_space = env.action_space(env_params)
    if isinstance(action_space, Box):
        action_shape = action_space.shape
        assert len(action_shape) == 1, "Can't handle action dim > 1"
        action_dim = action_shape[0]
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
    network = ActorCritic(action_dim,
                          is_continuous=is_continuous(action_space),
                          hidden_size=args.hidden_size,
                          activation=args.activation)
    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    _calculate_gae = calculate_gae

    def train(vf_coeff, lambda0, lr, rng):
        agent = PPO(network, vf_coeff=vf_coeff,
                    clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

        # initialize functions
        _env_step = partial(env_step, agent=agent, env=env, env_params=env_params, compute_hessian=False)
        _hessian_env_step = partial(env_step, agent=agent, env=env, env_params=env_params, compute_hessian=True)
        gae_lambda = jnp.array(lambda0)

        def hessian_computation(runner_state, epoch, at_init):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            reset_rng = jax.random.split(rng, args.num_envs)
            eval_obsv, eval_env_state = env.reset(reset_rng, env_params)
            init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
            eval_runner_state = (
                train_state,
                eval_env_state,
                eval_obsv,
                jnp.zeros((args.num_envs), dtype=bool),
                init_hstate,
                rng,
            )

            # COLLECT EVAL TRAJECTORIES
            eval_runner_state, eval_traj_batch = jax.lax.scan(
                _hessian_env_step, eval_runner_state, None, env_params.max_steps_in_episode
            )

            train_state, env_state, last_obs, last_done, hstate, rng = eval_runner_state
            _, _, last_val, _ = network.apply(train_state.params, hstate, (last_obs[np.newaxis, :], last_done[np.newaxis, :]))
            last_val = last_val.squeeze(0)
            advantages, targets = _calculate_gae(eval_traj_batch, last_val, last_done, gae_lambda, args.gamma)

            def hessian_loss(params, traj_batch, advantages, targets):
                total_loss, _ = agent.loss(params, init_hstate, traj_batch, advantages, targets)
                return total_loss
                            
            hvp_fn, unravel, num_params = get_hvp_fn(hessian_loss, train_state.params, (eval_traj_batch, advantages, targets))
            hvp_cl = lambda v: hvp_fn(train_state.params, v)
            rng, _rng = jax.random.split(rng)
            tridiag, lanczos_vecs = lanczos_alg(
                hvp_cl,
                num_params,
                order=100,
                rng_key=rng
            )   
            density, grids = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
            jax.debug.callback(plot_hessian_spectrum, grids, density, epoch, args.study_name, is_initialization=at_init)

        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
            )
            return lr * frac


        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, args.num_envs, *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, args.num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
        network_params = agent.network.init(_rng, init_hstate, init_x)
        param_count = sum(x.size for x in jax.tree_leaves(network_params))
        print('Network params number:', param_count)
        if args.anneal_lr:
            if args.weight_decay == 0.0:
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    adam_with_param_counts(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.add_decayed_weights(args.weight_decay),
                    adam_with_param_counts(learning_rate=linear_schedule, eps=1e-5),
                )
        else:
            if args.weight_decay == 0.0:
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    adam_with_param_counts(learning_rate=lr, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.add_decayed_weights(args.weight_decay),
                    adam_with_param_counts(learning_rate=lr, eps=1e-5),
                )
        
        tstate_class = TrainState
        if args.cont_backprop:
            tstate_class = ContinualBackpropTrainState
        train_state = tstate_class.create(
            apply_fn=agent.network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        # We first need to populate our LogEnvState stats.
        rng, _rng = jax.random.split(rng)
        init_rng = jax.random.split(_rng, args.num_envs)
        init_obsv, init_env_state = env.reset(init_rng, env_params)
        init_init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        init_runner_state = (
            train_state,
            env_state,
            init_obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_init_hstate,
            _rng,
        )

        starting_runner_state, _ = jax.lax.scan(
            _env_step, init_runner_state, None, env_params.max_steps_in_episode
        )

        def recursive_replace(env_state, new_env_state, names):
            if not isinstance(env_state, LogEnvState):
                return replace(env_state, env_state=recursive_replace(env_state.env_state, new_env_state.env_state, names))
            new_log_vals = {name: getattr(new_env_state, name) for name in names}
            return replace(env_state, **new_log_vals)

        replace_field_names = ['returned_episode_returns', 'returned_discounted_episode_returns', 'returned_episode_lengths']
        env_state = recursive_replace(env_state, starting_runner_state[1], replace_field_names)

        # TRAIN LOOP
        def _epoch_step(runner_state, unused):
            def _update_step(runner_state, unused):
                # COLLECT TRAJECTORIES
                initial_hstate = runner_state[-2]
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
                )

                # CALCULATE ADVANTAGE
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
                last_val = last_val.squeeze(0)

                advantages, targets = _calculate_gae(traj_batch, last_val, last_done, gae_lambda, args.gamma)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets = batch_info

                        grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, init_hstate, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state

                    # SHUFFLE COLLECTED BATCH
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, args.num_envs)
                    batch = (init_hstate, traj_batch, advantages, targets)

                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], args.num_minibatches, -1]
                                + list(x.shape[2:]),
                                ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss

                init_hstate = initial_hstate[None, :]  # TBH
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, args.update_epochs
                )
                train_state = update_state[0]

                # save metrics only every steps_log_freq
                metric = traj_batch.info
                metric = jax.tree.map(steps_filter, metric)

                rng = update_state[-1]
                if args.debug:

                    def callback(info):
                        timesteps = (
                                info["timestep"][info["returned_episode"]] * args.num_envs
                        )
                        if args.show_discounted:
                            show_str = "avg discounted return"
                            avg_return_values = jnp.mean(info["returned_discounted_episode_returns"][info["returned_episode"]])
                        else:
                            show_str = "avg episodic return"
                            avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])
                        friction = jnp.mean(info['friction'], axis=(0, 1, 2))[0]
                        if len(timesteps) > 0:
                            print(
                                f"timesteps={timesteps[0]} - {timesteps[-1]}, {show_str}={avg_return_values:.2f}, friction={friction}, "
                            )

                    jax.debug.callback(callback, metric)

                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)

                return runner_state, metric

            # returned metric has an extra dimension.
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, num_updates // num_epochs
            )

            metric = jax.tree.map(update_filter, metric)
            return runner_state, (metric, runner_state)

        rng, _rng = jax.random.split(rng)
        init_runner_state = (
            train_state, 
            env_state, 
            obsv, 
            jnp.zeros((args.num_envs), dtype=bool),
            init_hstate,
            _rng)
        # final_runner_state, (metric, all_runner_states) = jax.lax.scan(
        #     _epoch_step, init_runner_state, None, num_epochs
        # )
        runner_state = init_runner_state
        metrics_list = []
        states_list = []
        _epoch_step = jax.jit(_epoch_step)
        for epoch in range(num_epochs):
            if args.compute_hessian_init:
                hessian_computation(runner_state, epoch, at_init=True)
            runner_state, (metric, runner_state) = _epoch_step(runner_state, epoch)
            if args.compute_hessian_end:
                hessian_computation(runner_state, epoch, at_init=False)
            metrics_list.append(metric)
            states_list.append(runner_state)
        metric = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *metrics_list)
        all_runner_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states_list)
        final_runner_state = runner_state

        final_train_state = final_runner_state[0]
        runner_states = {
            'initial_runner_state': init_runner_state,
            'intermediate_runner_states': all_runner_states,
            'final_runner_state': final_runner_state
        }
        res = {"runner_state": runner_states, "metric": metric}
        return res

    return train


if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = NonStationaryPolicyHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)
    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    swept_args = deque()

    # we need to go backwards, since JAX returns indices
    # in the order in which they're vmapped.
    for i, arg in reversed(list(enumerate(train_args))):
        dims = [None] * len(train_args)
        dims[i] = 0
        vmaps_train = jax.vmap(vmaps_train, in_axes=dims)
        if arg == 'rng':
            swept_args.appendleft(rngs)
        else:
            assert hasattr(args, arg)
            swept_args.appendleft(getattr(args, arg))

    # train_jit = jax.jit(vmaps_train)
    train_jit = vmaps_train
    t = time()
    out = train_jit(*swept_args)
    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    final_train_state = out['runner_state']['final_runner_state'][0]
    if not args.save_runner_state:
        del out['runner_state']

    # our final_eval_metric returns max_num_steps.
    # we can filter that down by the max episode length amongst the runs.

    results_path = get_results_path(args, return_npy=False)

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state
    }

    all_results = jax.tree.map(numpyify, all_results)

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")