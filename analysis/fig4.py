import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import matplotlib.cm as cm
from pathlib import Path
import importlib.util

# Set up matplotlib for consistent styling
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

def smooth_data(data, window_size=4):
    """Smooth data using uniform_filter1d"""
    if window_size <= 1:
        return data
    
    if data.ndim == 1:
        return uniform_filter1d(data.astype(float), size=window_size, mode='nearest')
    else:
        smoothed = np.zeros_like(data, dtype=float)
        for seed_idx in range(data.shape[1]):
            smoothed[:, seed_idx] = uniform_filter1d(
                data[:, seed_idx].astype(float), size=window_size, mode='nearest')
        return smoothed

def plot_metric_in_ax(ax, all_reses, metric: str, title: str, is_rl_data: bool = False):
    """Plot data in a given axis without legend"""
    
    if is_rl_data:
        # RL environment plotting
        for study_name, res, color in all_reses:
            data = np.array(res[metric])
            n_seeds, n_steps = data.shape
            
            # Apply smoothing before computing statistics
            data_smoothed = smooth_data(data, window_size=4)
            env_mean = data_smoothed.mean(axis=0)
            env_std_err = sem(data_smoothed, axis=0)

            x_axis_multiplier = res['step_multiplier'][0]
            x = np.arange(env_mean.shape[0]) * x_axis_multiplier

            alpha_value = 1.0 if study_name == "L2 + ER" else 0.6
            ax.plot(x, env_mean, label=study_name, color=color, alpha=alpha_value)
            ax.fill_between(x, env_mean - env_std_err, env_mean + env_std_err,
                           color=color, alpha=alpha_value * 0.25)
            
        ax.set_xlabel('Environment steps', fontsize=48)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=48)
        
        # Make scientific notation larger on x-axis
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(24)
        
    else:
        # Continual learning plotting
        sample = all_reses[0][1]['outs'][metric]
        
        for study_name, res, color in all_reses:
            data = np.array(res['outs'][metric])
            
            # Handle data shapes
            num_tasks = sample.shape[-2] if len(sample.shape) > 2 else sample.shape[-1]
            x = np.arange(num_tasks)
            
            if len(sample.shape) > 2:
                data = data.reshape(data.shape[0], num_tasks, -1)
                data = np.sum(data, axis=-1)
            else:
                data = data.reshape(data.shape[0], num_tasks)
                
            means = data.mean(axis=0)
            errs = sem(data, axis=0) if data.shape[0] > 1 else np.zeros_like(means)
            
            if 'Continual ImageNet' in title:
                means = savgol_filter(means, window_length=20, polyorder=2)
 
            alpha_value = 1.0 if study_name == "L2 + ER" else 0.6
            ax.plot(x, means, label=study_name, color=color, alpha=alpha_value)
            ax.fill_between(x, means-errs, means+errs,
                           alpha=alpha_value * 0.25, color=color)

        ax.set_xlabel('Task', fontsize=48)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=48)
        
        if 'Incremental CIFAR' in title:
            ax.set_xticks(range(0, 20, 5))
    
    ax.tick_params(axis='both', which='major', labelsize=42)
    ax.set_title(title, fontsize=52, fontweight='bold')

def load_data(study_paths, is_rl=False):
    """Load data for given study paths"""
    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        
        # Add step multiplier for RL data only
        if is_rl and 'step_multiplier' not in best_res:
            if 'all_hyperparams' in best_res:
                step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
            else:
                hyperparams_dir = study_path.parent.parent / 'scripts' / 'hyperparams'
                study_hparam_filename = study_path.stem + '.py'
                hyperparam_path = hyperparams_dir / 'nonstationary' / 'hessian' / study_hparam_filename
                step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
            best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])
        
        all_reses.append((name, best_res, color))
    return all_reses

def get_total_steps_multiplier(saved_steps: int, hparam_path: Path):
    """Calculate step multiplier from hyperparameter files"""
    spec = importlib.util.spec_from_file_location('temp', hparam_path)
    var_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(var_module)
    all_list_hparams = getattr(var_module, 'hparams')['args']

    steps_multipliers = []
    for hparams in all_list_hparams:
        total_steps = hparams['total_steps']
        steps_multiplier = total_steps // saved_steps
        steps_multipliers.append(steps_multiplier)

    return steps_multipliers[0]

def create_combined_plot():
    """Create the 2x4 combined plot: left 4 blocks for dead_neurons, right 4 blocks for effective_rank"""
    fig, axes = plt.subplots(2, 4, figsize=(40, 18))
    
    # Define study paths
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    
    study_configs = {
        'mnist': [
            ('CBP', Path('/users/kguo32/rl-opt/permuted_mnist/results/cbp_hessian_fix_lr'), paired_colors[9]),
            ('L2 + ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_er_hessian_fix_lr'), paired_colors[1]),
            ('ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/er_hessian_fix_lr'), paired_colors[3]),
            ('BP', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp_hessian_fix_lr'), paired_colors[5]),
            ('L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_hessian_fix_lr'), paired_colors[7]),
        ],
        'imagenet': [
            ('CBP', Path('/users/kguo32/rl-opt/imagenet/results/cbp_hessian'), paired_colors[9]),
            ('L2 + ER', Path('/users/kguo32/rl-opt/imagenet/results/l2_er_hessian'), paired_colors[1]),
            ('ER', Path('/users/kguo32/rl-opt/imagenet/results/er_hessian'), paired_colors[3]),
            ('BP', Path('/users/kguo32/rl-opt/imagenet/results/bp_hessian'), paired_colors[5]),
            ('L2', Path('/users/kguo32/rl-opt/imagenet/results/l2_hessian'), paired_colors[7]),
        ],
        'cifar': [
            ('CBP', Path('/users/kguo32/rl-opt/incremental_cifar/results/cbp_hessian'), paired_colors[9]),
            ('L2 + ER', Path('/users/kguo32/rl-opt/incremental_cifar/results/l2_er_hessian'), paired_colors[1]),
            ('ER', Path('/users/kguo32/rl-opt/incremental_cifar/results/er_hessian'), paired_colors[3]),
            ('BP', Path('/users/kguo32/rl-opt/incremental_cifar/results/bp_hessian'), paired_colors[5]),
            ('L2', Path('/users/kguo32/rl-opt/incremental_cifar/results/l2_hessian'), paired_colors[7]),
            ('RESET', Path('/users/kguo32/rl-opt/incremental_cifar/results/reset_hessian'), 'black')
        ],
        'rl': [
            ('L2 + ER', Path('/users/kguo32/rl-opt/rlopt/results/l2_er_hessian'), paired_colors[1]),
            ('ER', Path('/users/kguo32/rl-opt/rlopt/results/er_hessian'), paired_colors[3]),
            ('L2', Path('/users/kguo32/rl-opt/rlopt/results/l2_hessian'), paired_colors[7]),
            ('BP', Path('/users/kguo32/rl-opt/rlopt/results/bp_hessian'), paired_colors[5]),
            ('CBP + L2', Path('/users/kguo32/rl-opt/rlopt/results/cbp_l2_hessian'), paired_colors[9]),
        ]
    }
    
    # Load data
    datasets = {
        'mnist': load_data(study_configs['mnist'], is_rl=False),
        'imagenet': load_data(study_configs['imagenet'], is_rl=False),
        'cifar': load_data(study_configs['cifar'], is_rl=False),
        'rl': load_data(study_configs['rl'], is_rl=True)
    }
    
    # Plot configurations - arrange by metric (left 4 blocks: dead_neurons, right 4 blocks: effective_rank)
    plot_configs = [
        # (row, col, dataset, metric, title, is_rl)
        # Left 4 blocks: Dead Neurons
        (0, 0, 'mnist', 'dead_neurons', 'Permuted MNIST', False),
        (0, 1, 'imagenet', 'dead_neurons', 'Continual ImageNet', False),
        (1, 0, 'cifar', 'dead_neurons', 'Incremental CIFAR', False),
        (1, 1, 'rl', 'dead_neurons', 'Slippery Ant', True),
        # Right 4 blocks: Effective Rank
        (0, 2, 'mnist', 'effective_rank', 'Permuted MNIST', False),
        (0, 3, 'imagenet', 'effective_rank', 'Continual ImageNet', False),
        (1, 2, 'cifar', 'effective_rank', 'Incremental CIFAR', False),
        (1, 3, 'rl', 'effective_rank', 'Slippery Ant', True),
    ]
    
    # Plot all subplots
    for row, col, dataset, metric, title, is_rl in plot_configs:
        plot_metric_in_ax(axes[row, col], datasets[dataset], metric, title, is_rl)
    
    # Create shared legend with independent handles
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Create new independent legend handles with thicker lines
    import matplotlib.lines as mlines
    legend_handles = []
    legend_labels = []
    
    # Get colors from original handles
    handle_colors = [handle.get_color() for handle in handles]
    handle_labels = [handle.get_label() for handle in handles]
    
    # Create new handles with thicker lines
    for color, label in zip(handle_colors, handle_labels):
        new_handle = mlines.Line2D([], [], color=color, label=label, linewidth=6)
        legend_handles.append(new_handle)
        legend_labels.append(label)
    
    # Add RESET if not present
    if 'RESET' not in legend_labels:
        reset_handle = mlines.Line2D([], [], color='black', label='RESET', linewidth=6)
        legend_handles.append(reset_handle)
        legend_labels.append('RESET')
    
    # Reorder legend to put "L2 + ER" first
    if 'L2 + ER' in legend_labels:
        l2_er_idx = legend_labels.index('L2 + ER')
        legend_labels.insert(0, legend_labels.pop(l2_er_idx))
        legend_handles.insert(0, legend_handles.pop(l2_er_idx))
    
    # Add legend at the bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=len(legend_labels), fontsize=48, frameon=True, fancybox=True, 
              shadow=True, borderpad=1.0, columnspacing=1.0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.5)
    
    return fig

if __name__ == "__main__":
    fig = create_combined_plot()
    
    # Save figures
    base_path = '/users/kguo32/rl-opt/analysis/dead_neurons_effective_rank'
    fig.savefig(f'{base_path}.pdf', bbox_inches='tight', dpi=300)
    print(f"Saved figures to {base_path}.pdf")