#!/bin/bash
#SBATCH --partition=3090-gcondo
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run_rl_cbp
#SBATCH --output=run_rl_cbp-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=run_rl_cbp-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ~/pobax_baseline/bin/activate
# run
python -m rlopt.ppo_nonstationary --env slippery_ant --total_steps 10000000 --num_envs 1 --weight_decay 0.001 --change_every 2000000 --compute_hessian_init --compute_hessian_end --lr 0.00025 --lambda0 0.95 --seed 2025 --n_seeds 5 --platform gpu --steps_log_freq 4 --update_log_freq 8 --debug --study_name l2_hessian