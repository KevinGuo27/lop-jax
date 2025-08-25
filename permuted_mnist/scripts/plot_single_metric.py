import argparse
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem

colors = {
    'pink': '#ff96b6',
    'red': '#df5b5d',
    'orange': '#DD8453',
    'yellow': '#f8de7c',
    'green': '#3FC57F',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'white': '#ffffff',
    'light gray': '#d5d5d5',
    'dark gray': '#666666',
    'black': '#000000'
}

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
        if metric in ['accuracy_eval', 'accuracy', 'accuracy_pre']:
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

        ax.plot(x, means, label=study_name, color=colors[color])
        ax.fill_between(x,
                        means-errs,
                        means+errs,
                        alpha=0.3,
                        color=colors[color])

    ax.set_xlabel('Task')
    ax.set_ylabel(metric.replace('_', ' ').capitalize())
    ax.legend()
    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', type=str,
                        help="Which output field to plot (e.g. 'accuracy_eval')")
    args = parser.parse_args()
    metric = args.metric

    env_name = 'permuted_mnist'
    study_paths = [
        # ('L2 + ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_er_fix_lr'), 'green'),
        # ('ER', Path('/users/kguo32/rl-opt/permuted_mnist/results/er_fix_lr'), 'cyan'),
        ('BP + LR(0.01)', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp_fix_lr'), 'blue'),
        ('BP + LR(0.001)', Path('/users/kguo32/rl-opt/permuted_mnist/results/bp'), 'cyan'),
        # ('L2', Path('/users/kguo32/rl-opt/permuted_mnist/results/l2_fix_lr'), 'yellow'),
        # ('CBP', Path('/users/kguo32/rl-opt/permuted_mnist/results/cbp_fix_lr'), 'red'),
    ]

    all_reses = []
    for name, study_path, color in study_paths:
        with open(study_path / "best_hyperparam_per_env_res.pkl", "rb") as f:
            best_res = pickle.load(f)
        all_reses.append((name, best_res, color))

    fig, ax = plot_reses(all_reses, metric=metric)

    # You could use plot_name if you want:
    plot_name = f"{env_name}_{metric}_bp_per_task.pdf"
    save_path = Path('/users/kguo32/rl-opt/permuted_mnist/results') / plot_name

    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")