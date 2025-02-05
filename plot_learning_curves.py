from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import orbax.checkpoint
from scipy.stats import sem, t

mpl.rcParams.update({
    # "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    # "font.sans-serif": ["Computer Modern Sans serif"],
    # "font.monospace": ["Computer Modern Typewriter"],
    "axes.labelsize": 25,  # LaTeX default is 10pt
    "font.size": 25,
    "legend.fontsize": 24,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
})

if __name__ == "__main__":

    # CBP
    ckpt_dirs = {
        'cbp': (Path("/Users/ruoyutao/Documents/rl-opt/results/slippery_ant_cbp/slippery_ant_ppo_seed(2024)_time(20241123-015247)_8d241715c04a80294c39f2b1adfb1c1d_np"), 'green'),
        # 'ppo': (Path("/Users/ruoyutao/Documents/rl-opt/results/slippery_ant/slippery_ant_ppo_seed(2024)_time(20241122-055452)_2b5761a2fa5b5664da2a7e9de0cbfd85_np"), 'orange')
    }

    confidence = 0.95

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    for name, (ckpt_dir, color) in ckpt_dirs.items():

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(ckpt_dir)
        args = restored['args']

        ep_returns = restored['metric']['returned_episode_returns'].squeeze()
        ep_returns = ep_returns.reshape(ep_returns.shape[0], -1)

        n_ticks = ep_returns.shape[-1]
        steps_per_tick = args['total_steps'] // n_ticks

        x = np.arange(n_ticks) * steps_per_tick
        for epr in ep_returns[:1]:
            ax.plot(x, epr, color=color, label=name)
        # mean_data = ep_returns.mean(axis=0)
        # std_err = sem(ep_returns, axis=0) * t.ppf((1 + confidence) / 2, mean_data.shape[0] - 1)
        #
        # ax.plot(x, mean_data, color=color, label=name)
        # ax.fill_between(x, mean_data - std_err, mean_data + std_err,
        #                 color=color, alpha=0.35)

    ax.set_ylim([-500, 1700])

    # now we add change_every lines
    n_changes = args['total_steps'] // args['change_every']
    change_ticks = np.arange(1, n_changes) * args['change_every']

    for ctick in change_ticks:
        ax.axvline(x=ctick, color='black', linestyle='--')

    plt.legend(loc='lower right')
    fig.tight_layout()

    plt.show()


    print()
