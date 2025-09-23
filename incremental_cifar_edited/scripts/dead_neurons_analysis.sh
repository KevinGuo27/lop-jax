#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=analyze_incremental_cifar
#SBATCH --output=post_run_cifar/post_run_cifar-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=post_run_cifar/post_run_cifar-%A.err

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ../../.venv/bin/activate
cd /users/tserapio/lop-jax/incremental_cifar/scripts

python process_all_bundles.py --results_base ../results --classes_per_task 5 --batch_size 1000 --dormant_unit_threshold 0.01 --agents bp
python process_all_bundles.py --results_base ../results --classes_per_task 5 --batch_size 1000 --dormant_unit_threshold 0.01 --agents l2_er
python process_all_bundles.py --results_base ../results --classes_per_task 5 --batch_size 1000 --dormant_unit_threshold 0.01 --agents l2
python process_all_bundles.py --results_base ../results --classes_per_task 5 --batch_size 1000 --dormant_unit_threshold 0.01 --agents cbp

# commented out, as i have no results for these agents yet
python process_all_bundles.py --results_base ../results --classes_per_task 5 --batch_size 1000 --dormant_unit_threshold 0.01 --agents er
python process_all_bundles.py --results_base ../results --classes_per_task 5 --batch_size 1000 --dormant_unit_threshold 0.01 --agents reset