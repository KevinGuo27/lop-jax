import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from pathlib import Path

# Set up matplotlib for consistent styling (exactly matching original scripts)
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

def plot_reses_in_ax(ax, all_reses, metric: str, title: str, font_scale: float = 1.7):
    """
    Plot data in a given axis without legend
    all_reses: list of (study_name, res_dict, color_key)
      - res_dict['outs'][metric] must be array shape (..., num_tasks) for continual learning
      - res_dict['scores'] for RL environments
    metric: name of the field to plot (e.g. 'accuracy_eval')
    """
    # Check if this is RL data (has 'scores') or continual learning data (has 'outs')
    is_rl_data = 'scores' in all_reses[0][1] and 'outs' not in all_reses[0][1]
    
    if is_rl_data:
        # RL environment plotting logic (exactly matching plot_best_hyperparam.py)
        for study_name, res, color in all_reses:
            scores = res['scores']  # Keep as original type
            
            # Following the exact logic from plot_best_hyperparam.py lines 81-127
            if isinstance(scores, list):
                mean_over_steps = [score.mean(axis=1)[..., 0] for score in scores]
                mean = [m.mean(axis=-1) for m in mean_over_steps]
                std_err = [sem(m, axis=-1) for m in mean_over_steps]
                n_seeds = mean_over_steps[0].shape[-1]
            else:
                # take mean over both
                mean_over_steps = scores.mean(axis=1)
                mean = mean_over_steps.mean(axis=-2)
                std_err = sem(mean_over_steps, axis=-2)
                n_seeds = mean_over_steps.shape[-2]
            
            # Select first environment (env_idx = 0)
            env_idx = 0
            if isinstance(mean, list):
                env_mean, env_std_err = mean[env_idx], std_err[env_idx]
            else:
                env_mean, env_std_err = mean[..., env_idx], std_err[..., env_idx]

            # Apply very light smoothing to RL data using Savitzky-Golay filter (better edge handling)
            if len(env_mean) >= 5:
                env_mean = savgol_filter(env_mean, window_length=3, polyorder=2)

            x_axis_multiplier = res['step_multiplier'][env_idx]
            x = np.arange(env_mean.shape[0]) * x_axis_multiplier

            # Use different transparency for L2-ER vs others
            alpha_value = 1.0 if study_name == "L2-ER" else 0.5
            ax.plot(x, env_mean, label=study_name, color=color, alpha=alpha_value)
            ax.fill_between(x, env_mean - env_std_err, env_mean + env_std_err,
                           color=color, alpha=alpha_value * 0.25)
            
        # Slightly larger font sizes for combined plot
        ax.set_xlabel('Environment steps', fontsize=int(28 * font_scale))
        ax.set_ylabel('Online returns', fontsize=int(28 * font_scale))
        ax.tick_params(axis='both', which='major', labelsize=int(24 * font_scale))
        
        # Make scientific notation larger on x-axis
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(int(20 * font_scale))
        
        # Set exact same y-axis limits as original
        ax.set_ylim(-2000, 4000)
    else:
        # Continual learning plotting logic
        sample = all_reses[0][1]['outs'][metric]
        
        for study_name, res, color in all_reses:
            data = np.array(res['outs'][metric])  # Convert to numpy for consistency
            
            # Handle different data shapes
            if metric in ['accuracy_eval', 'accuracy']:
                # Data shape is typically (n_seeds, ..., n_tasks)
                # Collapse all middle dimensions and keep last dimension as tasks
                if len(data.shape) == 5:  # e.g., (10, 1, 1, 1, 2000)
                    data = data.reshape(data.shape[0], -1)  # (10, 2000)
                elif len(data.shape) > 2:
                    # Collapse all but first and last dimensions
                    data = data.reshape(data.shape[0], -1)
                
                num_tasks = data.shape[-1]
                if title == 'Incremental CIFAR' and study_name in {"LayerNorm-L2", "Spectral Reg"}:
                    num_tasks = min(num_tasks, 19)
                    data = data[:, :num_tasks]
                x = np.arange(num_tasks)
            else:
                # For other metrics, assume second-to-last dimension is tasks
                num_tasks = data.shape[-2]
                x = np.arange(num_tasks)
                data = data.reshape(data.shape[0], num_tasks, -1)
                data = np.sum(data, axis=-1)  # Sum over last dimension
                
            means = data.mean(axis=0)
            errs = sem(data, axis=0) if data.shape[0] > 1 else np.zeros_like(means)
            
            # Apply smoothing only if we have enough points
            # Use larger smoothing window for ImageNet, smaller for others
            if len(means) >= 20 and title == 'Continual ImageNet':
                means = savgol_filter(means, window_length=30, polyorder=2)
                errs = savgol_filter(errs, window_length=30, polyorder=2)
            elif title != 'Incremental CIFAR':
                means = savgol_filter(means, window_length=5, polyorder=2)

            # Use different transparency for L2-ER vs others
            alpha_value = 1.0 if study_name == "L2-ER" else 0.6
            ax.plot(x, means, label=study_name, color=color, alpha=alpha_value)
            ax.fill_between(x, means-errs, means+errs,
                           alpha=alpha_value * 0.25, color=color)

        ax.set_xlabel('Task', fontsize=int(28 * font_scale))
        if metric in ['accuracy', 'accuracy_eval']:
            ax.set_ylabel('Accuracy', fontsize=int(28 * font_scale))
        else:
            ax.set_ylabel(metric.replace('_', ' ').capitalize(), fontsize=int(28 * font_scale))
        ax.tick_params(axis='both', which='major', labelsize=int(24 * font_scale))
        
        # Set specific x-axis formatting for Incremental CIFAR
        if title == 'Incremental CIFAR':
            ax.set_xticks(range(0, 20, 5))  # Ticks at 0, 5, 10, 15
    
    ax.set_title(title, fontsize=int(30 * font_scale), fontweight='bold')

def load_imagenet_data():
    """Load ImageNet data"""
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    study_paths = [
        ('CBP', Path('/users/kguo32/rl-opt/imagenet/results/cbp_hessian'), paired_colors[9]),
        ('L2-ER', Path('/users/kguo32/rl-opt/imagenet/results/l2_er_hessian'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/imagenet/results/er_hessian'), paired_colors[3]),
        ('BP', Path('/users/kguo32/rl-opt/imagenet/results/bp_hessian'), paired_colors[5]),
        ('L2', Path('/users/kguo32/rl-opt/imagenet/results/l2_hessian'), paired_colors[7]),
        ('LayerNorm-L2', Path('/users/kguo32/data/kguo32/lop/imagenet/results/laynorm_l2_hessian'), paired_colors[0]),
        ('Spectral Reg', Path('/users/kguo32/data/kguo32/lop/imagenet/results/spectral_reg_hessian'), paired_colors[11]),
        # ('L2 + Perturb', Path('/users/kguo32/rl-opt/imagenet/results/snp_l2'), paired_colors[11]),
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        all_reses.append((name, best_res, color))
    
    return all_reses

def load_cifar_data():
    """Load CIFAR data"""
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    study_paths = [
        ('CBP', Path('/users/kguo32/rl-opt/incremental_cifar/results/cbp_hessian'), paired_colors[9]),
        ('L2-ER', Path('/users/kguo32/rl-opt/incremental_cifar/results/l2_er_hessian'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/incremental_cifar/results/er_hessian'), paired_colors[3]),
        ('BP', Path('/users/kguo32/rl-opt/incremental_cifar/results/bp_hessian'), paired_colors[5]),
        ('L2', Path('/users/kguo32/rl-opt/incremental_cifar/results/l2_hessian'), paired_colors[7]),
        ('LayerNorm-L2', Path('/users/kguo32/rl-opt/incremental_cifar/results/layernorm_l2'), paired_colors[0]),
        ('Spectral Reg', Path('/users/kguo32/rl-opt/incremental_cifar/results/spectral_reg'), paired_colors[11]),
        ('RESET', Path('/users/kguo32/rl-opt/incremental_cifar/results/reset_hessian'), 'black')
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        all_reses.append((name, best_res, color))
    
    return all_reses

def load_mnist_data():
    """Load MNIST data"""
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    study_paths = [
        ('CBP', Path('/users/kguo32/rl-opt/permuted_mnist/results/cbp_hessian_fix_lr'), paired_colors[9]),
        ('L2-ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_er_hessian_fix_lr'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/er_hessian_fix_lr'), paired_colors[3]),
        ('BP', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp_hessian_fix_lr'), paired_colors[5]),
        ('L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_hessian_fix_lr'), paired_colors[7]),
        ('LayerNorm-L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/laynorm_l2_hessian_fix_lr'), paired_colors[0]),
        ('Spectral', Path('/users/kguo32/rl-opt/permuted_mnist/results/spectral_reg_hessian_fix_lr'), paired_colors[11]),
        # ('SNP + L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/snp_l2'), paired_colors[11]),
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        all_reses.append((name, best_res, color))
    
    return all_reses

def get_total_steps_multiplier(saved_steps: int, hparam_path: Path):
    """Calculate step multiplier from hyperparameter files (from plot_best_hyperparam.py)"""
    import importlib.util
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

def load_rl_data():
    """Load RL (Slippery Ant) data"""
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    study_paths = [
        ('L2-ER', Path('/users/kguo32/rl-opt/rlopt/results/l2_er'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/rlopt/results/er'), paired_colors[3]),
        ('L2', Path('/users/kguo32/rl-opt/rlopt/results/l2'), paired_colors[7]),
        ('BP', Path('/users/kguo32/rl-opt/rlopt/results/bp'), paired_colors[5]),
        ('CBP + L2', Path('/users/kguo32/rl-opt/rlopt/results/cbp_l2'), paired_colors[9]),
        ('LayerNorm-L2', Path('/users/kguo32/rl-opt/rlopt/results/layernorm_l2'), paired_colors[0]),
        ('Spectral Reg', Path('/users/kguo32/rl-opt/rlopt/results/spectral_reg'), paired_colors[11]),
        # ('Perturb + L2', Path('/users/kguo32/rl-opt/rlopt/results/snp_l2'), paired_colors[11]),
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        fname = "best_hyperparam_per_env_res.pkl"
        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)
        
        # Calculate step multiplier exactly like plot_best_hyperparam.py
        if 'all_hyperparams' in best_res:
            step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
        else:
            hyperparams_dir = study_path.parent.parent / 'scripts' / 'hyperparams'
            study_hparam_filename = study_path.stem + '.py'
            hyperparam_path = hyperparams_dir / 'nonstationary' / study_hparam_filename
            step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
        best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])
        
        all_reses.append((name, best_res, color))
    
    return all_reses

def create_combined_plot():
    """Create the 1x4 combined plot"""
    fig, axes = plt.subplots(1, 4, figsize=(40, 8))
    font_scale = 1.6
    
    # Load data for all four experiments
    mnist_data = load_mnist_data()
    imagenet_data = load_imagenet_data()
    cifar_data = load_cifar_data() 
    rl_data = load_rl_data()
    
    # Plot each subplot in the requested order: MNIST, ImageNet, CIFAR, Slippery Ant
    plot_reses_in_ax(axes[0], mnist_data, 'accuracy', 'Permuted MNIST', font_scale=font_scale)
    plot_reses_in_ax(axes[1], imagenet_data, 'accuracy_eval', 'Continual ImageNet', font_scale=font_scale)
    plot_reses_in_ax(axes[2], cifar_data, 'accuracy_eval', 'Incremental CIFAR', font_scale=font_scale)
    plot_reses_in_ax(axes[3], rl_data, 'scores', 'Slippery Ant', font_scale=font_scale)
    
    # Create a shared legend
    # Get the labels from the first subplot (they should be consistent across all)
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Add RESET from CIFAR data if not already present
    if 'RESET' not in labels:
        for study_name, res, color in cifar_data:
            if study_name == 'RESET':
                # Create a handle for RESET using the same style as other plots
                import matplotlib.lines as mlines
                reset_handle = mlines.Line2D([], [], color=color, label='RESET', linewidth=3)
                handles.append(reset_handle)
                labels.append('RESET')
                break
    
    # Reorder legend to put "L2-ER" first
    if 'L2-ER' in labels:
        l2_er_idx = labels.index('L2-ER')
        # Move L2-ER to the front
        labels.insert(0, labels.pop(l2_er_idx))
        handles.insert(0, handles.pop(l2_er_idx))
    
    for handle in handles:
        handle.set_linewidth(3)
    
    # Add the legend to the figure (positioned outside the subplots)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.27), 
              ncol=len(labels), fontsize=int(26 * font_scale), frameon=True, fancybox=True, 
              shadow=True, borderpad=1.0, columnspacing=1.0)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.3)
    
    return fig

if __name__ == "__main__":
    # Create the combined plot
    fig = create_combined_plot()
    
    # Save the figure
    save_path = Path('/users/kguo32/rl-opt/analysis/performance_comparison_2x2.pdf')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved combined figure to {save_path}")