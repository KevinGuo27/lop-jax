#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=l2_er_incremental_cifar
#SBATCH --output=l2er_logs_cifar/l2_er-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=l2er_logs_cifar/l2_er-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ../.venv/bin/activate
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
# run
python -m incremental_cifar.train_incremental_cifar --agent l2_er --lr 0.01 --er_lr 0.001 --weight_decay 0.001 --seed 2025 --n_seeds 1 --platform gpu --debug --study_name l2_er
# python -m incremental_cifar.train_incremental_cifar --agent l2_er --lr 0.01 --er_lr 0.001 --weight_decay 0.001 --seed 2027 --n_seeds 1 --platform gpu --debug --study_name l2_er
python -m incremental_cifar.train_incremental_cifar --agent l2_er --lr 0.01 --er_lr 0.001 --weight_decay 0.001 --seed 2028 --n_seeds 1 --platform gpu --debug --study_name l2_er