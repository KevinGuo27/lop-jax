from pathlib import Path

import numpy as np
import orbax.checkpoint
from scipy.interpolate import griddata


if __name__ == "__main__":
    key = 'cbp'
    ckpt_dir = Path('/Users/ruoyutao/Documents/rl-opt/results/slippery_ant_random_vecs/collected_dataset_20241209-124902')

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt_dir)
    taus = restored['taus']

    restored = restored[key]

    traj_batches, vals = restored['dataset']

    # Size of this array is (n_taus * n_taus) x n_ckpts x n_batches x episodes_per_batch x max_episodes_steps
    disc_returns = traj_batches['info']['returned_discounted_episode_returns']

    # TODO: get first returned episode. Doesn't matter here b/c Ant is essentially continuous
    # Get returns for first episode
    disc_return = disc_returns[..., -1]

    # we take the mean over n_episodes
    mean_disc_returns = disc_return.mean(axis=-1).mean(axis=-1)

    n_ckpts = mean_disc_returns.shape[-1]
    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

    grid_return_mean = griddata(taus, mean_disc_returns, (grid_x, grid_y), method='cubic')

    

    print()
