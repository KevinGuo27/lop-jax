import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter
import matplotlib.cm as cm
from pathlib import Path

# Set up matplotlib for consistent styling (exactly matching original scripts)
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

def load_permuted_mnist_data():
    """Load Permuted MNIST data from different optimization methods"""
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    study_paths = [
        ('CBP', Path('/users/kguo32/rl-opt/permuted_mnist/results/cbp'), paired_colors[9]),
        ('L2-ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_er'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/er'), paired_colors[3]),
        ('BP', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp'), paired_colors[5]),
        ('L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2'), paired_colors[7]),
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        all_reses.append((name, best_res, color))
    
    return all_reses

def plot_metric_in_ax(ax, all_reses, metric: str, title: str):
    """
    Plot a specific metric in the given axis
    
    Args:
        ax: matplotlib axis
        all_reses: list of (study_name, res_dict, color_key)
        metric: metric name ('accuracy', 'dead_neurons', 'effective_rank')
        title: plot title
    """
    
    for study_name, res, color in all_reses:
        data = np.array(res['outs'][metric])  # Shape: (5, 1, 1, 1, 800) or (5, 1, 1, 1, 800, 3)
        
        if metric in ['accuracy', 'accuracy_eval']:
            # For accuracy: shape is (5, 1, 1, 1, 800) -> reshape to (5, 800)
            data = data.reshape(data.shape[0], -1)  # (5, 800)
        elif metric in ['dead_neurons', 'effective_rank']:
            # For dead_neurons/effective_rank: shape is (5, 1, 1, 1, 800, 3)
            # Take mean over the last dimension (layers) and reshape
            data = data.mean(axis=-1)  # Average over layers: (5, 1, 1, 1, 800)
            data = data.reshape(data.shape[0], -1)  # (5, 800)
        
        num_tasks = data.shape[-1]
        x = np.arange(num_tasks)
        
        # Calculate mean and standard error across seeds
        means = data.mean(axis=0)
        errs = sem(data, axis=0) if data.shape[0] > 1 else np.zeros_like(means)
        
        # Apply light smoothing
        if len(means) >= 5:
            means = savgol_filter(means, window_length=5, polyorder=2)
            errs = savgol_filter(errs, window_length=5, polyorder=2)
        
        # Use different transparency for L2-ER vs others
        alpha_value = 1.0 if study_name == "L2-ER" else 0.6
        ax.plot(x, means, label=study_name, color=color, alpha=alpha_value, linewidth=2)
        ax.fill_between(x, means-errs, means+errs,
                       alpha=alpha_value * 0.25, color=color)

    # Set labels and formatting
    ax.set_xlabel('Task', fontsize=28)
    
    if metric in ['accuracy', 'accuracy_eval']:
        ax.set_ylabel('Accuracy', fontsize=28)
    elif metric == 'dead_neurons':
        ax.set_ylabel('Dead Neurons', fontsize=28)
    elif metric == 'effective_rank':
        ax.set_ylabel('Effective Rank', fontsize=28)
    
    ax.set_title(title, fontsize=30, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)

def create_fig7():
    """Create figure 7: 1x3 plot showing accuracy, dead neurons, and effective rank"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Load data
    mnist_data = load_permuted_mnist_data()
    
    # Plot each metric
    plot_metric_in_ax(axes[0], mnist_data, 'accuracy', 'Performance')
    plot_metric_in_ax(axes[1], mnist_data, 'dead_neurons', 'Dead Neurons')
    plot_metric_in_ax(axes[2], mnist_data, 'effective_rank', 'Effective Rank')
    
    # Create a shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Reorder legend to put "L2-ER" first
    if 'L2-ER' in labels:
        l2_er_idx = labels.index('L2-ER')
        # Move L2-ER to the front
        labels.insert(0, labels.pop(l2_er_idx))
        handles.insert(0, handles.pop(l2_er_idx))
    
    # Make legend lines thicker
    for handle in handles:
        handle.set_linewidth(3)
    
    # Add the legend to the figure (positioned below the subplots)
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.05), 
              ncol=len(labels), fontsize=26, frameon=True, fancybox=True, 
              shadow=True, borderpad=1.0, columnspacing=1.5)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    return fig

if __name__ == "__main__":
    # Create the figure
    fig = create_fig7()
    
    # Save the figure
    save_path = Path('/users/kguo32/rl-opt/analysis/permuted_mnist_best.pdf')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure 7 to {save_path}")
    
    # Also show the plot
    plt.show()
