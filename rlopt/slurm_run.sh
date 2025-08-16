#!/bin/bash
#SBATCH --partition=3090-gcondo
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run_slippery_ant
#SBATCH --output=run_slippery_ant-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=run_slippery_ant-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ~/pobax_baseline/bin/activate
# run
python -m rlopt.ppo_nonstationary --env slippery_ant --total_steps 2000000 --num_envs 1 --num_minibatches 16 --num_steps 2048 --vf_coeff 1.0 --replacement_rate 0.0001 --maturity_threshold 10000 --weight_decay 0.0 --change_every 2000000 --cont_backprop --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --lambda0 0.95 --seed 2025 --n_seeds 5 --platform gpu --steps_log_freq 4 --update_log_freq 8 --debug --study_name cbp