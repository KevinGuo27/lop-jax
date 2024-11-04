from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.stats import sem, t

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

def conf_int(arr: np.ndarray, axis: int = -1, confidence: float = 0.95):
    se = sem(interpolated_discounted_returns, axis=axis)
    return se * t.ppf((1 + confidence) / 2, arr.shape[axis] - 1)


if __name__ == "__main__":
    test_dir = Path(ROOT_DIR, 'results', 'cartpole_ppo')
    res_dir_1 = test_dir / 'CartPole-v1_ppo_seed(2024)_time(20241017-161347)_0804fc50930714b8fd36ebdb48bc5e43_policy_eval_results'
    res_dir_2 = test_dir / 'CartPole-v1_ppo_seed(2024)_time(20241017-161946)_638b1e247933c0c14cfa9c79c2da65d7_policy_eval_results'

    # test_dir = Path(ROOT_DIR, 'results', 'ant')
    # res_dir_1 = test_dir / 'ant_seed(2024)_time(20240927-050217)_839c382d193eb2af6573ea4faa63dc4a_policy_eval_results'
    # res_dir_2 = test_dir / 'ant_seed(2024)_time(20240927-164410)_da55c1cbf98ef5681833d1c3d41db8e8_policy_eval_results'
    res_infos = [
        (res_dir_1, 'Baseline', 'orange'),
        (res_dir_2, 'L2 Reg', 'green')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for i, (res_path, title, color) in enumerate(res_infos):
        ax = axes[i]

        res = np.load(res_path / 'parsed_results.npy', allow_pickle=True).item()
        t_args = res['train_args']
        l2_reg_val = t_args['l2_reg_coeff'].item()
        title = title + f' ({l2_reg_val})'

        taus = res['taus']

        interpolated_discounted_returns = res['interpolated_discounted_returns']
        interpolated_returns = res['interpolated_returns']
        if t_args['alg'] == 'ppo':
            init_gae = res['init_gae']
            mean_measure = init_gae.mean(axis=-1)
            std_err_measure = conf_int(init_gae, axis=-1)
        else:
            mean_measure = interpolated_discounted_returns.mean(axis=-1)
            std_err_measure = conf_int(interpolated_discounted_returns, axis=-1)

        # mean_returns = interpolated_returns.mean(axis=-1)
        # sem_returns = sem(interpolated_returns, axis=-1)

        ax.plot(taus, mean_measure, color=colors[color])

        ax.fill_between(taus, mean_measure - std_err_measure, mean_measure + std_err_measure,
                        color=colors[color], alpha=0.35)
        ax.set_ylim([20, 90])
        # ax.set_ylim([-1.5, 1.5])
        ax.set_title(title)

        if i == 0:
            if t_args['alg'] == 'ppo':
                ax.set_ylabel('GAE')
            else:
                ax.set_ylabel('Discounted Return')
        ax.set_xlabel('tau')

        # TODO: plot returns on top of deez
    plt.show()


