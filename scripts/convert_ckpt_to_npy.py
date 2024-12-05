from pathlib import Path

from flax.training import orbax_utils
import jax
import orbax.checkpoint

from rlopt.utils import numpyify

if __name__ == "__main__":

    ckpt_dir = Path("/home/taodav/Documents/rl-opt/results/slippery_ant/slippery_ant_ppo_seed(2024)_time(20241122-055452)_2b5761a2fa5b5664da2a7e9de0cbfd85")
    # ckpt_dir = Path("/home/taodav/Documents/rl-opt/results/slippery_ant_cbp/slippery_ant_ppo_seed(2024)_time(20241123-015247)_8d241715c04a80294c39f2b1adfb1c1d")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt_dir)

    restored_np = jax.tree.map(numpyify, restored)
    ckpt_dir_np = ckpt_dir.parent / (ckpt_dir.stem + '_np')

    save_args = orbax_utils.save_args_from_target(restored_np)

    print(f"Saving results to {ckpt_dir_np}")
    orbax_checkpointer.save(ckpt_dir_np, restored_np, save_args=save_args)
