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
# python train_imagenet.py --debug --agent l2_er --num_tasks 100 --lr 0.01 --weight_decay 1e-4 --er_lr 0.001 --er_batch 12 --compute_hessian --compute_hessian_interval 10 --compute_hessian_size 2000
python train_imagenet.py --debug --agent bp --num_tasks 100 --lr 0.01 --weight_decay 0.0