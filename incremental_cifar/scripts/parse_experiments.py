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
from permuted_mnist.utils.file_system import load_info

def parse_exp_dir(study_path, study_hparam_path):
    # Hyperparameters of interest
    train_sign_hparams = ['lr', 'weight_decay', 'replacement_rate', 'er_lr', 'spectral_reg_strength']

    # Gather all result paths, excluding previous summary
    study_paths = list(Path(study_path).iterdir())
    study_paths = [p for p in study_paths if not p.name.endswith('best_hyperparam_per_env_res.pkl')]

    # Containers for metrics, outputs, and hyperparams
    eval_dict = {}
    out_dict = {}
    hyperparams = {}

    # Load each experiment's results
    for results_path in tqdm(study_paths):
        results_path = Path(results_path).resolve()
        if results_path.suffix == '.npy':
            restored = load_info(results_path)
        else:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(results_path)

        args = restored['args']
        # Key by (lr, weight_decay)
        args_tuple = tuple(
            float(v.item()) if hasattr(v, 'item') else float(v)
            for v in (args[hp] for hp in train_sign_hparams)
        )

        out = restored['out']
        # accuracy_eval shape: (er_lr, seed, tasks)
        online_eval = out['accuracy_eval']

        # Accumulate per-key metrics and full outputs
        eval_dict.setdefault(args_tuple, []).append(online_eval)
        out_dict.setdefault(args_tuple, []).append(out)
        hyperparams[args_tuple] = args

    # Stack metrics across seeds
    scores = {k: np.stack(vs, axis=0) for k, vs in eval_dict.items()}

    # Combine outputs across seeds: stack each field along a new axis
    combined_outs = {}
    for k, outs_list in out_dict.items():
        # assume all outs have the same keys
        fields = outs_list[0].keys()
        combined = {}
        for field in fields:
            combined[field] = np.stack([d[field] for d in outs_list], axis=0)
        combined_outs[k] = combined

    # Select best hyperparameter setting by mean score
    max_mean_score = -np.inf
    best_args = None
    best_score = None
    for args_tuple, score in scores.items():
        mean_score = score.mean()
        if mean_score > max_mean_score:
            max_mean_score = mean_score
            best_score = score
            best_args = args_tuple

    best_hyperparams = hyperparams[best_args]
    best_outs = combined_outs[best_args]

    print(f"Best hyperparams: {best_hyperparams}")
    print(f"Max score: {max_mean_score}")
    return {
        'scores': best_score,
        'hyperparams': best_hyperparams,
        'trained_hyperparams': train_sign_hparams,
        'outs': best_outs,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study_path', type=str)
    args = parser.parse_args()

    study_path = Path(args.study_path)
    study_hparam_path = Path(
        '/users/kguo32/rl-opt/permuted_mnist/scripts/hyperparams',
        study_path.stem + '.py'
    )

    parsed_res = parse_exp_dir(study_path, study_hparam_path)
    parsed_res_path = study_path / 'best_hyperparam_per_env_res.pkl'

    print(f"Saving parsed results to {parsed_res_path}")
    with open(parsed_res_path, 'wb') as f:
        pickle.dump(parsed_res, f, protocol=4)