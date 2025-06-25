#!/bin/bash
#SBATCH --partition=3090-gcondo
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run_permuted_mnist
#SBATCH --output=run_permuted_mnist-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=run_permuted_mnist-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ~/pobax_baseline/bin/activate
# run
python train_pi_params_nonstationary.py --env slippery_ant --debug --change_every 2000000 --num_steps 128 --num_minibatches 4 --lr 0.0001 --hidden_size 256 --total_steps 10000000 --platform gpu --cont_backprop --study_name slippery_ant_cbp --no_anneal_lr --n_param_sets 1 --compute_hessian_init