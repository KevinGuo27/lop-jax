from functools import partial
import time
from pathlib import Path

import chex
from flax.training import orbax_utils
from gymnax.environments.spaces import Box, Discrete
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import orbax.checkpoint

from rlopt.agents import ActorCriticAgent, PPOAgent
from rlopt.config import PolicyHyperparams
from rlopt.cbp import generate_seeds_for_pytree
from rlopt.envs import load_nonstationary_env, is_continuous
from rlopt.models import ActorCritic
from rlopt.policy_eval import policy_eval
from rlopt.utils import numpyify

from definitions import ROOT_DIR


def get_random_params(rng: chex.PRNGKey, params: dict):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's params, so one direction entry per weight.
    """
    param_rngs = generate_seeds_for_pytree(rng, params)

    def rand_norm_of_shape(x: jnp.ndarray, rng: chex.PRNGKey):
        return jax.random.normal(rng, shape=x.shape)

    return jax.tree.map(rand_norm_of_shape, params, param_rngs)


def normalize_direction(direction: dict, params: dict, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          params: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        def filter_norm(d, w):
            return d * (jnp.linalg.norm(w) / (jnp.linalg.norm(d) + 1e-10))

        return jax.tree.map(filter_norm, direction, params)
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        raise NotImplementedError
        # direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        raise NotImplementedError
        # direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        raise NotImplementedError
        # for d in direction:
        #     d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        raise NotImplementedError
        # direction.div_(direction.norm())
    else:
        raise NotImplementedError


def create_random_direction(rng: chex.PRNGKey, params: dict,
                            ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    direction = get_random_params(rng, params)

    # potentially fill biases with 0
    def maybe_fill_bias(d: jnp.ndarray, p: jnp.ndarray):
        if len(d.shape) <= 1:
            if ignore == 'biasbn':
                return jnp.zeros_like(d)
            else:
                return p
        return d

    direction = jax.tree.map(maybe_fill_bias, direction, params)
    normalized_direction = normalize_direction(direction, params, norm=norm)

    return normalized_direction


def augment_params_two_directions(params: dict, dir1: dict, dir2: dict,
                                  steps: jnp.ndarray):

    def augment(p: jnp.ndarray, d1: jnp.ndarray, d2: jnp.ndarray):
        changes = d1 * steps[0] + d2 * steps[1]
        return p + changes

    return jax.tree.map(augment, params, dir1, dir2)


def dstack_product(x, y):
    return jnp.dstack(jnp.meshgrid(x, y)).reshape(-1, 2)


if __name__ == "__main__":
    seed = 2024
    n_eval_episodes = 1
    # n_eval_episodes = 10
    episodes_per_batch = 1
    # episodes_per_batch = 10

    tau_start = -1
    tau_end = 1
    # DEBUGGING
    n_taus = 3
    # n_taus = 20
    product_n_taus = int(n_taus**2)

    tau_array = jnp.linspace(tau_start, tau_end, num=n_taus)
    taus = dstack_product(tau_array, tau_array)

    seed_idx = 0

    # CBP
    ckpt_dirs = {
        'cbp': Path(ROOT_DIR, "results/slippery_ant_cbp/slippery_ant_ppo_seed(2024)_time(20241123-015247)_8d241715c04a80294c39f2b1adfb1c1d_np"),
        # 'ppo': Path("/Users/ruoyutao/Documents/rl-opt/results/slippery_ant/slippery_ant_ppo_seed(2024)_time(20241122-055452)_2b5761a2fa5b5664da2a7e9de0cbfd85_np")
    }

    rng = jax.random.PRNGKey(seed)

    res_dict = {}

    for k, fpath in ckpt_dirs.items():
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(fpath)
        args = restored['args']
        args = PolicyHyperparams().from_dict(args)

        intermediate_ts, intermediate_env_state, _, _ = restored['runner_states']['intermediate_runner_states']

        # these all have leading dimension n_seeds x n_epochs, we index into seed_idx
        all_params = jax.tree.map(lambda x: x[seed_idx], intermediate_ts['params'])
        all_frictions = intermediate_env_state['env_state']['info']['sys_variation']['geom_friction'][seed_idx, :, 0, 0, 0]

        ep_returns = restored['metric']['returned_episode_returns'].squeeze()
        ep_returns = ep_returns.reshape(ep_returns.shape[0], -1)[seed_idx]

        rng, dir_rng = jax.random.split(rng)

        vmap_augment_params_two_directions = jax.vmap(augment_params_two_directions, in_axes=[None, None, None, 0])

        def create_two_random_directions_and_augment(rng, params):
            dir1_rng, dir2_rng = jax.random.split(rng)
            dir1 = create_random_direction(dir1_rng, params)
            dir2 = create_random_direction(dir2_rng, params)

            augmented_params = vmap_augment_params_two_directions(params, dir1, dir2, taus)

            return augmented_params

        vmap_create_two_random_directions = jax.vmap(create_two_random_directions_and_augment, in_axes=0)
        n_params = all_frictions.shape[0]
        dir_rngs = jax.random.split(dir_rng, n_params)
        augmented_params = vmap_create_two_random_directions(dir_rngs, all_params)  # has leading dims n_epochs x (n_taus * n_taurs)

        # switch to len(taus) x n_epochs, b/c we want to iterate over taus
        augmented_params = jax.tree.map(lambda x: x.swapaxes(0, 1), augmented_params)

        # Now we do policy eval for all of these bad bois
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

        if args.alg == 'ppo':
            agent = PPOAgent(network, args)
        else:
            agent = ActorCriticAgent(network, args)

        policy_eval_fn = jax.jit(partial(policy_eval, env=env, env_params=env_params,
                                 agent=agent, n_episodes=n_eval_episodes,
                                 episodes_per_batch=episodes_per_batch))

        rng, reset_rng = jax.random.split(rng)
        reset_rng = reset_rng[None, :]
        obsv, env_state = env.reset(reset_rng, env_params)

        obsv = obsv[0]
        env_state = jax.tree.map(lambda x: x[0], env_state)

        vmapped_pe_fn = jax.vmap(policy_eval_fn, in_axes=[None, None, 0, 0])

        # TODO: scan w/ tqdm
        @jax.jit
        @scan_tqdm(product_n_taus)
        def pe_iteration(rng, x):
            i, inp = x
            env_state, obs, pi_params = inp
            rng, _rng = jax.random.split(rng)
            _rngs = jax.random.split(_rng, n_params)
            pe_out = vmapped_pe_fn(env_state, obs, _rngs, pi_params)
            return rng, pe_out

        rng, pe_rng = jax.random.split(rng)
        env_states = jax.tree.map(lambda x: x[None, ...].repeat(product_n_taus, axis=0), env_state)
        obses = jax.tree.map(lambda x: x[None, ...].repeat(product_n_taus, axis=0), obsv)
        inp = (env_states, obses, augmented_params)

        _, dataset = jax.lax.scan(
            pe_iteration, pe_rng, (jnp.arange(product_n_taus), inp), product_n_taus
        )

        res_dict[k] = {'dataset': dataset, 'path': fpath}


    res_study_dir = Path(ROOT_DIR, 'results', f'{args.env}_random_vecs')
    res_study_dir.mkdir(exist_ok=True)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    res_dir = res_study_dir / f'collected_dataset_{time_str}'

    res_dict = jax.tree.map(numpyify, res_dict)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(res_dict)

    print(f"Saving results to {res_dir}")
    orbax_checkpointer.save(res_dir, res_dict, save_args=save_args)



