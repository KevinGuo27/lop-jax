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
python train_permuted_mnist.py --debug
