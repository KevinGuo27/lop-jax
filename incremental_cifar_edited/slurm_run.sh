#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run_incremental_cifar
#SBATCH --output=run_incremental_cifar-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=run_incremental_cifar-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ../.venv/bin/activate
# run
python train_incremental_cifar.py --debug --agent bp --weight_decay 0.0 --num_features 1000 --compute_hessian --compute_hessian_interval 40 --compute_hessian_size 2000