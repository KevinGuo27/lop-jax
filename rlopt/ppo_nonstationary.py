from typing import NamedTuple

from collections import deque
from dataclasses import replace
from functools import partial
import inspect
from time import time
import pickle
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
from rlopt.models import ActorCritic
from rlopt.utils.optimizer import adam_with_param_counts
from rlopt.utils.evaluation import summarize_all_layers
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
    activations: jnp.ndarray
    info: jnp.ndarray = None

class PPO:
    def __init__(self, network,
                 vf_coeff: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2,
                 use_spectral_reg: bool = False,
                 spectral_k: int = 2,
                 spectral_target: float = 2.0,
                 spectral_reg_strength: float = 0.1,
                 spectral_power_iter: int = 10):
        self.network = network
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps
        self.use_spectral_reg = use_spectral_reg
        self.spectral_k = spectral_k
        self.spectral_target = spectral_target
        self.spectral_reg_strength = spectral_reg_strength
        self.spectral_power_iter = spectral_power_iter
        self.act = jax.jit(self.act)
        self.loss = jax.jit(self.loss)
    
    def perturb(self, params, perturb_scale, rng):
        """Add Gaussian noise to parameters"""
        return perturb_params(params, rng, perturb_scale)

    def act(self, rng: chex.PRNGKey,
            train_state: flax.training.train_state.TrainState,
            obs: chex.Array):

        # SELECT ACTION
        pi, value, activations = self.network.apply(train_state.params, obs)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return value, action, log_prob, activations
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, obs):
        _, _, activations = self.network.apply(params, obs)
        pol_erank_losses = [self.effective_rank(f) for f in activations['actor'].values() if f is not None]
        val_erank_losses = [self.effective_rank(f) for f in activations['critic'].values() if f is not None]
        loss_erank = - (jnp.stack(pol_erank_losses).mean() +
                        jnp.stack(val_erank_losses).mean())
        return loss_erank

    def loss(self, params, traj_batch, gae, targets):
        # RERUN NETWORK
        pi, value, _ = self.network.apply(
            params, traj_batch.obs
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
        if self.use_spectral_reg:
            spectral_reg = compute_spectral_regularization(
                params,
                k=self.spectral_k,
                target=self.spectral_target,
                reg_strength=self.spectral_reg_strength,
                num_iter=self.spectral_power_iter,
            )
            total_loss = total_loss + spectral_reg
        return total_loss, (value_loss, loss_actor, entropy)


def perturb_params(params, rng, scale):
    """Add N(0, scale) noise to layer parameters (weights and biases) only."""
    
    def perturb_layer_params(layer_params, rng):
        """Perturb weights and biases of a single layer."""
        rng1, rng2 = jax.random.split(rng)
        perturbed_kernel = layer_params['kernel'] + scale * jax.random.normal(
            rng1, layer_params['kernel'].shape, layer_params['kernel'].dtype)
        perturbed_bias = layer_params['bias'] + scale * jax.random.normal(
            rng2, layer_params['bias'].shape, layer_params['bias'].dtype)
        return {'kernel': perturbed_kernel, 'bias': perturbed_bias}
    
    actor_layer_keys = [k for k in params['params']['actor'].keys() if k.startswith('a_')]
    critic_layer_keys = [k for k in params['params']['critic'].keys() if k.startswith('c_')]
    num_actor_layers = len(actor_layer_keys)
    num_critic_layers = len(critic_layer_keys)
    # Perturb actor layers
    rngs = jax.random.split(rng, num_actor_layers)
    layer_idx = 0
    for key, value in params['params']['actor'].items():
        if key.startswith('a_'):
            params['params']['actor'][key] = perturb_layer_params(value, rngs[layer_idx])
            layer_idx += 1
    # Perturb critic layers
    rngs = jax.random.split(rng, num_critic_layers)
    layer_idx = 0
    for key, value in params['params']['critic'].items():
        if key.startswith('c_'):
            params['params']['critic'][key] = perturb_layer_params(value, rngs[layer_idx])
            layer_idx += 1
    
    return params, rngs[-1]


def power_iteration(w, num_iter=10, rng_key=None):
    """Compute largest singular value using power iteration."""
    eps = 1e-6
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    v = jax.random.normal(rng_key, (w.shape[1], ))

    for _ in range(num_iter):
        wv = jnp.matmul(w, v)
        v = jnp.matmul(w.T, wv)
        v = v / (jnp.linalg.norm(v) + eps)

    Av = jnp.matmul(w, v)
    s = jnp.linalg.norm(Av)
    u = Av / (s + eps)

    return s, u, v


def compute_spectral_regularization(params, k=2, target=2.0, reg_strength=0.1, num_iter=10):
    """Compute spectral regularization on network parameters."""
    reg = 0.0
    param_idx = 0

    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        # Extract the parameter name from the path
        if len(path) >= 2:
            param_type = path[-1].key if hasattr(path[-1], 'key') else str(path[-1])

            if param_type == 'kernel':
                # Use deterministic key based on parameter index to avoid randomness in gradients
                rng_key = jax.random.PRNGKey(param_idx)
                largest_sv, _, _ = power_iteration(leaf, num_iter=num_iter, rng_key=rng_key)
                spectral_reg = (largest_sv**k - target)**2
                reg += reg_strength * spectral_reg
                param_idx += 1
            elif param_type == 'bias':
                reg += reg_strength * jnp.sum((leaf - 0.0)**2)
            elif param_type == 'scale':
                reg += reg_strength * jnp.sum((leaf - 1.0)**2)

    return reg


def env_step(runner_state, unused, agent: PPO, env, env_params, args):
    train_state, env_state, last_obs, rng = runner_state
    rng, _rng = jax.random.split(rng)
    value, action, log_prob, activations = agent.act(_rng, train_state, last_obs)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, args.num_envs)
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    
    transition = Transition(
        done, action, value, reward, log_prob, last_obs, activations, info
    )
    runner_state = (train_state, env_state, obsv, rng)
    return runner_state, transition


def calculate_gae(traj_batch, last_val, gae_lambda, gamma):
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
    print('num_epochs:', num_epochs)
    num_friction_changes = (
        args.total_steps // args.change_every // args.num_envs
    )
    with open("/users/kguo32/rl-opt/rlopt/frictions", 'rb+') as f:
        frictions = pickle.load(f)
    friction_schedule = frictions[args.friction_seed][:num_friction_changes]
    print('Friction schedule:', friction_schedule)
    env, env_params = load_nonstationary_env(rand_key, args.env, gamma=args.gamma, 
                        change_every=args.change_every, friction_schedule=friction_schedule)

    print('Observation space:', env.observation_space(env_params))
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
    network = ActorCritic(
        action_dim,
        is_continuous=is_continuous(action_space),
        hidden_size=args.hidden_size,
        activation=args.activation,
        use_layernorm=args.use_layernorm,
    )
    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    _calculate_gae = calculate_gae

    def train(vf_coeff, lambda0, er_lr, lr, rng):
        agent = PPO(
            network,
            vf_coeff=vf_coeff,
            clip_eps=args.clip_eps,
            entropy_coeff=args.entropy_coeff,
            use_spectral_reg=args.use_spectral_reg,
            spectral_k=args.spectral_k,
            spectral_target=args.spectral_target,
            spectral_reg_strength=args.spectral_reg_strength,
            spectral_power_iter=args.spectral_power_iter,
        )

        # initialize functions
        _env_step = partial(env_step, agent=agent, env=env, env_params=env_params, args=args)
        _hessian_env_step = partial(env_step, agent=agent, env=env, env_params=env_params, args=args)

        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
            )
            return lr * frac

        def hessian_computation(runner_state, epoch, at_init):
            train_state, env_state, last_obs, rng = runner_state
            reset_rng = jax.random.split(rng, args.num_envs)
            eval_obsv, eval_env_state = env.reset(reset_rng, env_params)
            eval_runner_state = (
                train_state,
                eval_env_state,
                eval_obsv,
                rng,
            )

            # COLLECT EVAL TRAJECTORIES
            eval_runner_state, eval_traj_batch = jax.lax.scan(
                _hessian_env_step, eval_runner_state, None, env_params.max_steps_in_episode
            )

            train_state, env_state, last_obs, rng = eval_runner_state
            _, last_val, _ = network.apply(train_state.params, last_obs)

            advantages, targets = _calculate_gae(eval_traj_batch, last_val, lambda0, args.gamma)

            def hessian_loss(params, traj_batch, advantages, targets):
                total_loss, _ = agent.loss(params, traj_batch, advantages, targets)
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
            jax.debug.callback(plot_hessian_spectrum, grids, density, epoch, args.study_name, at_init=at_init)

        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = agent.network.init(_rng, init_x)
        param_count = sum(x.size for x in jax.tree_leaves(network_params))
        print('Network params number:', param_count)
        if args.anneal_lr:
            if args.optimizer == 'sgd':
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.add_decayed_weights(args.weight_decay),
                    optax.sgd(learning_rate=linear_schedule),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.add_decayed_weights(args.weight_decay),
                    adam_with_param_counts(learning_rate=linear_schedule, b1=args.beta_1, b2=args.beta_2),
                )
        else:
            if args.optimizer == 'sgd':
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.add_decayed_weights(args.weight_decay),
                    optax.sgd(learning_rate=lr),
                )
            elif args.optimizer == 'muon':
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.contrib.muon(learning_rate=lr, adam_b1=args.beta_1, adam_b2=args.beta_2, weight_decay=args.weight_decay),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.add_decayed_weights(args.weight_decay),
                    adam_with_param_counts(learning_rate=lr, b1=args.beta_1, b2=args.beta_2),
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

        # We first need to populate our LogEnvState stats.
        rng, _rng = jax.random.split(rng)
        init_rng = jax.random.split(_rng, args.num_envs)
        init_obsv, init_env_state = env.reset(init_rng, env_params)

        init_runner_state = (
            train_state,
            env_state,
            init_obsv,
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
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
                )

                # CALCULATE ADVANTAGE
                train_state, env_state, last_obs, rng = runner_state
                _, last_val, _ = network.apply(train_state.params, last_obs)

                advantages, targets = _calculate_gae(traj_batch, last_val, lambda0, args.gamma)

                # CALCULATE DEAD NEURONS AND EFFECTIVE RANK
                features_list = [f for f in traj_batch.activations['actor'].values() if f is not None] + \
                                [f for f in traj_batch.activations['critic'].values() if f is not None]
                rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features_list)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_erbatch(train_state_rng, er_batches):
                        def _update_erank(runner_state, unused):
                            train_state, er_batches, rng = runner_state
                            traj_batch, advantages, targets = er_batches
                            er_loss = agent.effective_rank_loss(train_state.params, traj_batch.obs)
                            grads = jax.grad(agent.effective_rank_loss)(train_state.params, traj_batch.obs)
                            updates = jax.tree_util.tree_map(lambda g: -er_lr * g, grads)
                            new_params = optax.apply_updates(train_state.params, updates)
                            train_state = train_state.replace(params=new_params)
                            return (train_state, er_batches, rng), er_loss

                        def _update_minbatch(train_state_rng, batch_info):
                            train_state, rng = train_state_rng
                            traj_batch, advantages, targets = batch_info

                            grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                            total_loss, grads = grad_fn(
                                train_state.params, traj_batch, advantages, targets
                            )
                            train_state = train_state.apply_gradients(grads=grads)
                            
                            # Apply perturb if enabled
                            if args.to_perturb:
                                rng, _rng = jax.random.split(rng)
                                new_params, rng = agent.perturb(train_state.params, args.perturb_scale, _rng)
                                train_state = train_state.replace(params=new_params)
                            
                            if args.cont_backprop:
                                rng, _rng = jax.random.split(rng)
                                train_state = train_state.update_and_reinit(_rng,
                                                                            traj_batch.activations,
                                                                            replacement_rate=args.replacement_rate,
                                                                            decay_rate=args.decay_rate,
                                                                            maturity_threshold=args.maturity_threshold)
                            return (train_state, rng), total_loss

                        assert args.num_minibatches % args.er_batch == 0, "Effective rank batch size must be divisible by num_minibatches"
                        minibatches = jax.tree_util.tree_map(
                            lambda x: jnp.reshape(
                                x,
                                [args.er_batch, -1]
                                + list(x.shape[1:]),
                            ),
                            er_batches,
                        )
                        train_state, rng = train_state_rng
                        if args.er:
                            er_runner_state = (train_state, er_batches, rng)
                            er_runner_state, er_loss = jax.lax.scan(_update_erank, er_runner_state, None, args.er_step)
                            train_state, _, rng = er_runner_state

                        (train_state, rng), total_loss = jax.lax.scan(
                            _update_minbatch, (train_state, rng), minibatches
                        )
                        return (train_state, rng), total_loss

                    (
                        train_state,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state

                    # SHUFFLE COLLECTED BATCH
                    rng, _rng = jax.random.split(rng)
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
                    er_batches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x,
                            [args.num_minibatches // args.er_batch, -1]
                            + list(x.shape[1:]),
                        ),
                        shuffled_batch,
                    )
                    (train_state, rng), total_loss = jax.lax.scan(
                        _update_erbatch, (train_state, rng), er_batches
                    )

                    update_state = (
                        train_state,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss

                update_state = (
                    train_state,
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
                metric['dead_neurons'] = dead_neurons
                metric['effective_rank'] = effective_rank

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
                        dead_neurons = info['dead_neurons']
                        effective_rank = info['effective_rank']
                        if len(timesteps) > 0:
                            print(
                                f"timesteps={timesteps[0]} - {timesteps[-1]}, {show_str}={avg_return_values:.2f}, friction={friction}, dead neurons={dead_neurons}, effective rank={effective_rank}"
                            )

                    jax.debug.callback(callback, metric)
                    # jax.debug.print("dead neurons: {dead_neurons}, effective rank: {effective_rank}", dead_neurons=dead_neurons, effective_rank=effective_rank)

                runner_state = (train_state, env_state, last_obs, rng)

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
            _rng)
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
        metric = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *metrics_list)
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

    return train, friction_schedule


if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = NonStationaryPolicyHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn, friction_schedule = make_train(args, make_train_rng)
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
        'final_train_state': final_train_state,
        'friction_schedule': friction_schedule,
    }

    all_results = jax.tree.map(numpyify, all_results)

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")