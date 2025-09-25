import importlib
import pickle
from pathlib import Path
import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.cm as cm
from scipy.ndimage import uniform_filter1d


from definitions import ROOT_DIR

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

# rc('text', usetex=True)

# colors = {
#     'pink': '#ff96b6',
#     'red': '#df5b5d',
#     'orange': '#DD8453',
#     'yellow': '#f8de7c',
#     'green': '#3FC57F',
#     'cyan': '#48dbe5',
#     'blue': '#3180df',
#     'purple': '#9d79cf',
#     'brown': '#886a2c',
#     'white': '#ffffff',
#     'light gray': '#d5d5d5',
#     'dark gray': '#666666',
#     'black': '#000000'
# }

env_name_to_title = {
    'rocksample_15_15': 'RockSample (15, 15)',
    'rocksample_11_11': 'RockSample (11, 11)',
    'pocman': 'Pocman',
    'battleship_10': 'Battleship 10x10',
    'battleship_5': 'Battleship 5x5',
    'cheese.95': 'Cheese',
    'hallway': 'Hallway',
    'heavenhell': 'Heavenhell',
    'network': 'Network',
    'paint': 'Paint',
    'shuttle': 'Shuttle',
    'tiger-alt-start': 'Tiger',
    'tmaze_5': 'T-maze'
}

env_name_to_x_upper_lim = {
    '4x3': 1e6,
    'cheese.95': 1e6,
    'hallway': 2e6,
    'network': 1e6,
    'paint': 1e6,
    'tiger-alt-start': 1e6,
    'tmaze_5': 2e6
}


def smooth_data(data, window_size=5, method='moving_average'):
    """
    Smooth data to reduce variance.
    
    Args:
        data: numpy array of shape (time_steps, n_seeds) or (time_steps,)
        window_size: size of the smoothing window
        method: 'moving_average' or 'exponential'
    
    Returns:
        smoothed data with same shape as input
    """
    if window_size <= 1:
        return data
    
    if data.ndim == 1:
        if method == 'moving_average':
            return uniform_filter1d(data.astype(float), size=window_size, mode='nearest')
        elif method == 'exponential':
            # Exponential smoothing
            alpha = 2.0 / (window_size + 1)
            result = np.zeros_like(data, dtype=float)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
    else:
        # Apply smoothing to each seed separately
        smoothed = np.zeros_like(data, dtype=float)
        for seed_idx in range(data.shape[1]):
            if method == 'moving_average':
                smoothed[:, seed_idx] = uniform_filter1d(
                    data[:, seed_idx].astype(float), size=window_size, mode='nearest')
            elif method == 'exponential':
                alpha = 2.0 / (window_size + 1)
                smoothed[0, seed_idx] = data[0, seed_idx]
                for i in range(1, data.shape[0]):
                    smoothed[i, seed_idx] = (alpha * data[i, seed_idx] + 
                                           (1 - alpha) * smoothed[i-1, seed_idx])
        return smoothed
    
    return data

def plot_reses(all_reses: list[tuple], metric, n_rows: int = 2,
               individual_runs: bool = False, smooth_window: int = 5, 
               smooth_method: str = 'moving_average'):
    # plt.rcParams.update({'font.size': 32})
    # check to see that all our envs are the same across all reses.
    for _, x, _ in all_reses:
        for i in range(len(x['envs'])):
            if x['envs'][i].endswith('pixels'):
                x['envs'][i] = x['envs'][i][:-7]
                print(x['envs'][i])
    all_envs = [set(x['envs']) for _, x, _ in all_reses]
    for envs in all_envs:
        assert envs == all_envs[0]

    envs = list(sorted(all_envs[0]))

    n_rows = min(n_rows, len(envs))
    n_cols = max((len(envs) + 1) // n_rows, 1) if len(envs) > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 5))

    for k, (study_name, res, color) in enumerate(all_reses):
        scores = res[metric]
        
        # Apply smoothing to each individual run before computing statistics
        if smooth_window > 1:
            scores_smoothed = smooth_data(scores, window_size=smooth_window, method=smooth_method)
        else:
            scores_smoothed = scores
            
        mean = scores_smoothed.mean(axis=-2)
        std_err = sem(scores_smoothed, axis=-2)

        for i, env in enumerate(envs):
            row = i // n_cols
            col = i % n_cols

            env_idx = res['envs'].index(env)
            env_mean, env_std_err = mean, std_err
            n_seeds = scores.shape[-2]

            x_axis_multiplier = res['step_multiplier'][env_idx]

            if len(envs) == 1:
                ax = axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[i]
            x = np.arange(env_mean.shape[0]) * x_axis_multiplier
            x_upper_lim = env_name_to_x_upper_lim.get(env, None)

            # Use different transparency for L2 + ER vs others
            alpha_value = 1.0 if study_name == "L2 + ER" else 0.6
            ax.plot(x, env_mean, label=study_name, color=color, alpha=alpha_value)
            ax.fill_between(x, env_mean - env_std_err, env_mean + env_std_err,
                                color=color, alpha=alpha_value * 0.25)
            ax.set_xlabel('Environment steps', fontsize=16)
            if metric in ['accuracy', 'accuracy_eval']:
                ax.set_ylabel('Accuracy', fontsize=16)
            else:
                ax.set_ylabel(metric.replace('_', ' ').capitalize(), fontsize=16)
            ax.legend()

    fig.tight_layout()

    plt.show()
    return fig, axes


def get_total_steps_multiplier(saved_steps: int, hparam_path: Path):
    spec = importlib.util.spec_from_file_location('temp', hparam_path)
    var_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(var_module)
    all_list_hparams = getattr(var_module, 'hparams')['args']

    steps_multipliers = []
    for hparams in all_list_hparams:
        assert 'total_steps' in hparams
        total_steps = hparams['total_steps']
        steps_multiplier = total_steps // saved_steps
        steps_multipliers.append(steps_multiplier)

    assert all(m == steps_multipliers[0] for m in steps_multipliers)
    return steps_multipliers[0]


def find_file_in_dir(file_name: str, base_dir: Path) -> Path:
    for path in base_dir.rglob('*'):
        if file_name in str(path):
            return path

if __name__ == "__main__":
    env_name = 'slippery_ant'
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', type=str,
                        help="Which output field to plot (e.g. 'accuracy_eval')")
    parser.add_argument('--smooth-window', type=int, default=5,
                        help="Window size for smoothing (default: 5, set to 1 to disable)")
    parser.add_argument('--smooth-method', type=str, default='moving_average',
                        choices=['moving_average', 'exponential'],
                        help="Smoothing method to use (default: moving_average)")
    args = parser.parse_args()
    metric = args.metric
    smooth_window = args.smooth_window
    smooth_method = args.smooth_method

    # Get set2 color palette
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    # normal
    study_paths = [
        ('L2 + ER', Path('/users/kguo32/rl-opt/rlopt/results/l2_er_hessian'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/rlopt/results/er_hessian'), paired_colors[3]),
        ('L2', Path('/users/kguo32/rl-opt/rlopt/results/l2_hessian'), paired_colors[7]),
        ('BP', Path('/users/kguo32/rl-opt/rlopt/results/bp_hessian'), paired_colors[5]),
        ('CBP + l2', Path('/users/kguo32/rl-opt/rlopt/results/cbp_l2_hessian'), paired_colors[9])
    ]

    hyperparam_type = 'per_env'  # (all_env | per_env)
    plot_name = f'{env_name}_{metric}_{hyperparam_type}'

    if study_paths[0][1].stem.endswith('best'):
        plot_name += '_best'
    
    if smooth_window > 1:
        plot_name += f'_smooth_{smooth_method}_{smooth_window}'

    envs = None

    all_reses = []

    for name, study_path, color in study_paths:
        if hyperparam_type == 'all_env':
            fname = "best_hyperparam_res.pkl"
            if name == 'PPO Markov':
                fname = 'best_hyperparam_res_F_split.pkl'
        elif hyperparam_type == 'per_env':
            fname = "best_hyperparam_per_env_res.pkl"

        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)

        all_reses.append((name, best_res, color))

        if 'all_hyperparams' in best_res:
            step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
        else:
            hyperparams_dir = study_path.parent.parent / 'scripts' / 'hyperparams'
            study_hparam_filename = study_path.stem + '.py'
            hyperparam_path = Path('hyperparams', 'nonstationary', 'hessian', study_hparam_filename)
            step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
        best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

    fig, axes = plot_reses(all_reses, metric, individual_runs=False, n_rows=3,
                          smooth_window=smooth_window, smooth_method=smooth_method)

    save_plot_to = Path('/users/kguo32/rl-opt/rlopt/results', f'{plot_name}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")