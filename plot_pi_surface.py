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
    # test_dir = Path(ROOT_DIR, 'results', 'cartpole')
    # res_dir_1 = test_dir / 'CartPole-v1_seed(2024)_time(20240923-095942)_91cc50897622a4b99dea8d29b285cd94_policy_eval_results'
    # res_dir_2 = test_dir / 'CartPole-v1_seed(2024)_time(20240923-100456)_6b30bcefa311f2ecc54a6522c01be8b2_policy_eval_results'

    test_dir = Path(ROOT_DIR, 'results', 'pendulum')
    res_dir_1 = test_dir / 'Pendulum-v1_seed(2024)_time(20240926-133317)_1325eb0e60d8a7501b67e2544ee1fde1_policy_eval_results'
    res_dir_2 = test_dir / 'Pendulum-v1_seed(2024)_time(20240926-133829)_2ac03c0fea318234f31061a9584b57c3_policy_eval_results'
    res_infos = [
        (res_dir_1, 'Baseline', 'orange'),
        (res_dir_2, 'L2 Reg', 'green')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for i, (res_path, title, color) in enumerate(res_infos):
        ax = axes[i]

        res = np.load(res_path / 'parsed_results.npy', allow_pickle=True).item()
        l2_reg_val = res['train_args'].item().l2_reg_coeff
        title = title + f' ({l2_reg_val})'

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
        # ax.set_ylim([19.5, 20.01])
        ax.set_ylim([-86, -72])
        ax.set_title(title)

        if i == 0:
            ax.set_ylabel('Discounted Return')
        ax.set_xlabel('tau')

        # TODO: plot returns on top of deez
    plt.show()


