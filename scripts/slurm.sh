#!/bin/bash
#SBATCH --partition=3090-gcondo
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run_permuted_mnist
#SBATCH --output=run_permuted_mnist-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=run_permuted_mnist-%A.err

module load cuda cudnn

pyenv activate jaxopt

python permuted_mnist/train_permuted_mnist.py --agent er --debug True