#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda cudnn

# Initialize pyenv - source bashrc to get pyenv in PATH
source ~/.bashrc

# Activate the environment
pyenv activate jaxopt

python permuted_mnist/train_permuted_mnist.py --agent bp --weight_decay 0.0 --debug --num_features 1000 --lr 0.003