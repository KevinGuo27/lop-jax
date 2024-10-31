from functools import partial
import inspect
from pathlib import Path
from typing import Sequence, Union

import chex
import distrax
import flax.linen as nn
from flax.linen.initializers import glorot_normal, glorot_uniform
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from gymnax.environments.spaces import Box, Discrete
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from optax import tree_utils as otu

from rlopt.agents import ActorCriticAgent, PPOAgent, Transition
from rlopt.config import PolicyHyperparams
from rlopt.envs import load_env, is_continuous
from rlopt.file_system import get_results_path
from rlopt.models import ActorCritic

jax.config.update("jax_disable_jit", True)


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]

def ContinualLearningState(NamedTuple):
  """State for the Continual Learning algorithm."""
  utility: optax._src.base.Updates
  age: optax._src.base.Updates
  eligible_neurons: float


def continual_backprop(
        rho: float = 1e-5,
        eta: float = 0.99,
        maturity_thresh: float = 100,
        mu_dtype: Optional[Any] = None,
) -> optax.base.GradientTransformation:

    def init_fn(params):
        utility = otu.tree_zeros_like(params, dtype=jnp.float32)
        age = otu.tree_zeros_like(params, dtype=jnp.float32)
        eligible_neurons = 0.0
        return ContinualLearningState(utility=utility, eligible_neurons=eligible_neurons)

    def update_fn(updates, state,  params, intermediates):
        #utility = updates.
        def is_leaf_sum_batch(node):
            if type(node) != dict and type(node) == tuple:
                if len(node) == 1 and getattr(node[0], "shape", None) is not None:
                    node_shape = node[0].shape
                    if len(node_shape) == 2:
                        return True
            return False

        batch_average_intermediates = jax.tree.map(
            lambda batched_outputs: jnp.average(jnp.abs(batched_outputs[0]), axis=0),
            intermediates,
            is_leaf=is_leaf_sum_batch
        )

        def is_leaf_broadcast_activation(node):
            valid_node_keys = False
            if type(node) == dict:
                valid_node_keys = True
                all_node_keys = node.keys()
                for node_key in all_node_keys:
                    if node_key not in ["__call__", "bias", "kernel"]:
                        valid_node_keys = False
                        break
            return valid_node_keys

        def replace_with_utility(activation_leaf_dict, params_leaf_dict):
            activation_val = activation_leaf_dict["__call__"][0]
            all_vals = jnp.sum(jnp.abs(jax.tree.leaves(params_leaf_dict)))
            total_utility = activation_val*all_vals
            tree_updated = jax.tree_map(lambda x: total_utility, params_leaf_dict)
            return tree_updated

        broadcasted_intermediates = jax.tree_map(replace_with_utility, batch_average_intermediates, params,
                                                 is_leaf=is_leaf_broadcast_activation)

        state.utility = jax.tree.map(lambda x, y: eta*x + (1 - eta)*y, state.utility, broadcasted_intermediates)
        state.age = jax.tree.map(lambda x: x + 1, state.age)

        def return_count_if_valid(node_dict):
            if node_dict["bias"][0] > m:
                return 1
            return 0

        new_eligibility = jax.tree_util.tree_reduce(
            lambda x, y: x["bias"][0] + jnp.sum(y) if x["bias"][0] > maturity_thresh else 0,
            state.age, initializer=0, is_leaf=is_leaf_broadcast_activation)

        state.eligibile = state.eligibile + rho*new_eligibility
        eligibility_int = int(state.eligibile)

        utility_leaves = jax.tree_util.tree_leaves(state.utility)

        # Concatenate all leaf values into a single array
        concatenated_utility = jnp.concatenate([jnp.ravel(leaf) for leaf in utility_leaves])

        # Get unique and sorted values
        if eligibility_int > len(concatenated_utility):
            eligibility_int = len(concatenated_utility)
        upper_bound_eligible = jnp.sort(jnp.unique(concatenated_utility))[eligibility_int - 1]

        replacement_multiplicative_term = jax.tree.map(
            lambda param_util, param_age: -1.0 if param_util <= upper_bound_eligible and param_age >= maturity_thresh else 0.0,
        state.utility, state.age
        )






    return optax.base.GradientTransformation(init_fn, update_fn)

class ActorCriticOG(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=glorot_normal()
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=glorot_normal()
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=glorot_uniform()
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=glorot_normal()
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=glorot_normal()
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=glorot_uniform())(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def make_train(rng: chex.PRNGKey, args: PolicyHyperparams):
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )
    minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )
    env, env_params = load_env(args.env, gamma=args.gamma)

    action_space = env.action_space(env_params)
    if isinstance(action_space, Box):
        action_shape = action_space.shape
        assert len(action_shape) == 1, "Can't handle action dim > 1"
        action_dim = action_shape[0]
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
    network = ActorCriticOG(is_continuous=is_continuous(action_space),
                          action_dim=action_dim,
                          hidden_size=args.hidden_size)

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_x)

    if args.alg == 'ppo':
        print("PPO agent bruv")
        agent = PPOAgent(network, args)
    else:
        print("NOT PPO agent bruv")
        agent = ActorCriticAgent(network, args)

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (args.num_minibatches * args.update_epochs))
                / num_updates
        )
        return args.lr * frac

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    updates_filter = partial(filter_period_first_dim, n=args.updates_log_freq)

    def train(rng):
        if args.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                value, action, log_prob = agent.act(_rng, train_state.params, last_obs)

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
            _, last_val = network.apply(train_state.params, last_obs)
            advantages, targets = agent.target(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, return_intermediates=True
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    import pdbr
                    pdbr.set_trace()
                    return train_state, total_loss

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
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
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

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        metric = jax.tree.map(updates_filter,
                              metric)  # update_steps x (args.num_steps // args.steps_log_freq) x num_envs

        return {"runner_state": runner_state, "metrics": metric}

    return train


def run_train(passed_in_args: Union[dict, PolicyHyperparams] = None) -> Path:
    jax.disable_jit(True)
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
