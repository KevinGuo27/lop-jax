import argparse
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.cm as cm

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

def plot_reses(all_reses, metric: str):
    """
    all_reses: list of (study_name, res_dict, color_key)
      - res_dict['outs'][metric] must be array shape (..., num_tasks)
    metric: name of the field to plot (e.g. 'accuracy_eval')
    """
    # determine number of tasks
    sample = all_reses[0][1]['outs'][metric]

    fig, ax = plt.subplots(figsize=(8, 5))
    for study_name, res, color in all_reses:
        data = res['outs'][metric]
        # collapse all but last axis
        if metric in ['accuracy']:
            num_tasks = sample.shape[-1]
            x = np.arange(num_tasks)
            data = data.reshape(-1, num_tasks)
        else:
            num_tasks = sample.shape[-2]
            x = np.arange(num_tasks)
            data = data.reshape(-1, num_tasks, data.shape[-1])
            data = np.sum(data, axis=-1)
        means = data.mean(axis=0)
        errs  = sem(data, axis=0)

        # Use different transparency for L2 + ER vs others
        alpha_value = 1.0 if study_name == "L2 + ER" else 0.6
        ax.plot(x, means, label=study_name, color=color, alpha=alpha_value)
        ax.fill_between(x,
                        means-errs,
                        means+errs,
                        alpha=alpha_value * 0.25,  # Make fill_between more transparent
                        color=color)

    ax.set_xlabel('Task', fontsize=24)
    if metric in ['accuracy', 'accuracy_eval']:
        ax.set_ylabel('Accuracy', fontsize=24)
    else:
        ax.set_ylabel(metric.replace('_', ' ').capitalize(), fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Larger legend text
    ax.legend(fontsize=20)
    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', type=str,
                        help="Which output field to plot (e.g. 'accuracy_eval')")
    args = parser.parse_args()
    metric = args.metric
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']

    env_name = 'permuted_mnist'
    # Get set2 color palette
    paired_colors = cm.Paired(np.linspace(0, 1, 12))
    study_paths = [
        ('CBP', Path('/users/kguo32/rl-opt/permuted_mnist/results/cbp_hessian_fix_lr'), paired_colors[9]),
        ('L2 + ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_er_hessian_fix_lr'), paired_colors[1]),
        ('ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/er_hessian_fix_lr'), paired_colors[3]),
        ('BP', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp_hessian_fix_lr'), paired_colors[5]),
        ('L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_hessian_fix_lr'), paired_colors[7]),
        ('LayerNorm', Path('/users/kguo32/rl-opt/permuted_mnist/results/laynorm_l2'), paired_colors[0]),
        ('Spectral Reg', Path('/users/kguo32/rl-opt/permuted_mnist/results/spectral_reg'), paired_colors[11]),
        # ('BP_MSE', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp_mse'), paired_colors[5]),
        # ('SNP + L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/snp_l2'), paired_colors[11]),
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        all_reses.append((name, best_res, color))

    fig, ax = plot_reses(all_reses, metric=metric)

    # You could use plot_name if you want:
    plot_name = f"{env_name}_{metric}_hessian_per_task.pdf"
    save_path = Path('/users/kguo32/rl-opt/permuted_mnist/results') / plot_name

    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")