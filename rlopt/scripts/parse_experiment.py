import argparse
from collections import OrderedDict
import importlib
from pathlib import Path
import pickle
import re
import sys

import jax
import jax.numpy as jnp
import orbax.checkpoint
import numpy as np
from tqdm import tqdm

from definitions import ROOT_DIR
from pobax.utils.file_system import load_info

def combine_seeds_and_envs(x: jnp.ndarray):
    # Here, dim=-1 is the NUM_ENVS parameter. We take the mean over this.
    # dim=-2 is the NUM_STEPS parameter.
    # dim=-3 is the NUM_UPDATES, which is TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS.
    # dim=-4 is n_seeds.
    # We take the mean and std_err to the mean over dimensions -1 and -4.
    x = jnp.array(x)
    envs_seeds_swapped = jnp.swapaxes(x, -2, -4).swapaxes(-3, -4)
    print(f"Env seeds swapped shape: {envs_seeds_swapped.shape}")

    # We take the mean over NUM_ENVS dimension.
    mean_over_num_envs = envs_seeds_swapped.mean(axis=-1)
    return mean_over_num_envs

def parse_exp_dir(study_path, study_hparam_path):
    # TODO: THIS
    train_sign_hparams = ['vf_coeff', 'lambda0', 'lr', 'weight_decay']
    study_paths = list(study_path.iterdir())
    # remove the path that ends with best_hyperparam_per_env_res.pkl
    study_paths = [path for path in study_paths if not path.name.endswith('best_hyperparam_per_env_res.pkl')]

    scores, final_scores, envs, hyperparams, eval_dict, final_eval_dict = {}, {}, [], {}, {}, {}
    for results_path in tqdm(study_paths):
        results_path = Path(results_path).resolve()
        if results_path.suffix == '.npy':
            restored = load_info(results_path)
        else:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(results_path)

        args = restored['args']
        args_tuple = tuple(
            float(v.item()) if hasattr(v, "item") else float(v)
            for v in (args[hp] for hp in train_sign_hparams)
        )
        # Get online metrics
        online_eval = restored['out']['metric']
        online_disc_returns = online_eval['returned_episode_returns']
        # online disc returns has shape (num_updates // update_frequency, num_steps // step_frequency, n_envs)
        if args_tuple in eval_dict:
            eval_dict[args_tuple].append(online_disc_returns)
        else:
            eval_dict[args_tuple] = [online_disc_returns]
        hyperparams[args_tuple] = args
        del restored

    
    # combine the seeds
    for args_tuple, online_disc_returns in eval_dict.items():
        eval_dict[args_tuple] = np.stack(online_disc_returns, axis=0)

    for args_tuple, online_disc_returns in eval_dict.items(): 
        seeds_combined = combine_seeds_and_envs(online_disc_returns)
        scores[args_tuple] = seeds_combined
        # Scores has shape (num_updates // update_frequency, num_steps // step_frequency, seed = 1)
        # final_scores has shape (1, 1, seed = 1)

    # Find the best hyperparameters
    # TODO
    max_mean_score = -np.inf
    best_hyperparams = None
    max_score = None

    for args_tuple, score in scores.items():
        score = score.squeeze()  # remove the last dimension if it is 1
        print(score.shape)
        mean_score = score.mean(axis=-1).mean(axis=-1).mean(axis=-1)
        if mean_score > max_mean_score:
            max_mean_score = mean_score
            best_hyperparams = hyperparams[args_tuple]
            max_score = score
    print(f"Best hyperparams: {best_hyperparams}")
    envs.append(best_hyperparams['env'])
    max_score = jnp.expand_dims(max_score, axis=-1)

    parsed_res = {
        'envs': envs,
        'scores': max_score,
        'hyperparams': best_hyperparams,
        'trained_hyperparams': train_sign_hparams,
    }
    return parsed_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('study_path', type=str)
    args = parser.parse_args()

    study_path = Path(args.study_path)
    study_hparam_path = Path('/users/kguo32/rl-opt/rlopt/scripts/hyperparams/nonstationary', study_path.stem + '.py')

    parsed_res_path = study_path / "best_hyperparam_per_env_res.pkl"

    parsed_res = parse_exp_dir(study_path, study_hparam_path)

    print(f"Saving parsed results to {parsed_res_path}")
    with open(parsed_res_path, 'wb') as f:
        pickle.dump(parsed_res, f, protocol=4)