from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.stats import sem

from definitions import ROOT_DIR

rc('text', usetex=True)
# rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('font', **{'size': 32})
rc('axes', unicode_minus=False)


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

if __name__ == "__main__":
    test_dir = Path(ROOT_DIR, 'results', 'test')
    res_dir_1 = test_dir / 'CartPole-v1_seed(2024)_time(20240921-120623)_7da5238bd1736ccc406ae1851b84d22f_policy_eval_results'
    res_dir_2 = test_dir / 'CartPole-v1_seed(2024)_time(20240923-072252)_388972e42d810da4b1ee6dacc73e2ee5_policy_eval_results'
    res_infos = [
        (res_dir_1, 'Baseline', 'orange'),
        (res_dir_2, 'L2 Reg', 'green')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for i, (res_path, title, color) in enumerate(res_infos):
        ax = axes[i]

        res = np.load(res_path / 'parsed_results.npy', allow_pickle=True).item()

        taus = res['taus']

        interpolated_discounted_returns = res['interpolated_discounted_returns']
        interpolated_returns = res['interpolated_returns']

        mean_disc_returns = interpolated_discounted_returns.mean(axis=-1)
        sem_disc_returns = sem(interpolated_discounted_returns, axis=-1)

        mean_returns = interpolated_returns.mean(axis=-1)
        sem_returns = sem(interpolated_returns, axis=-1)

        ax.plot(taus, mean_disc_returns, color=colors[color])

        ax.fill_between(taus, mean_disc_returns - sem_disc_returns, mean_disc_returns + sem_disc_returns,
                        color=colors[color], alpha=0.35)
        ax.set_ylim([19.7, 20.01])
        ax.set_title(title)
        if i == 0:
            ax.set_ylabel('Discounted Return')
        ax.set_xlabel('tau')
    plt.show()


