#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=cbp_incremental_cifar
#SBATCH --output=cbp_logs_cifar/cbp-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=cbp_logs_cifar/cbp-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ../.venv/bin/activate
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
# run
python -m incremental_cifar.train_incremental_cifar --agent cbp --cont_backprop --weight_decay 0.0 --lr 0.01 --replacement_rate 1e-06 --seed 2025 --n_seeds 1 --platform gpu --debug --study_name cbp
python -m incremental_cifar.train_incremental_cifar --agent cbp --cont_backprop --weight_decay 0.0 --lr 0.01 --replacement_rate 1e-06 --seed 2027 --n_seeds 1 --platform gpu --debug --study_name cbp
python -m incremental_cifar.train_incremental_cifar --agent cbp --cont_backprop --weight_decay 0.0 --lr 0.01 --replacement_rate 1e-06 --seed 2028 --n_seeds 1 --platform gpu --debug --study_name cbp