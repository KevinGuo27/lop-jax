#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --job-name=lin_seeds
#SBATCH --output=slurm_logs/lin_permuted_mnist-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=slurm_logs/lin_permuted_mnist-%A.err

# activate your environment
module load python/3.11

# Generate timestamped study folder name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
STUDY_NAME="linearized_seeds_${TIMESTAMP}"
echo "Results will be saved to: results/${STUDY_NAME}/"

for seed in 0 1 2 3 4; do

    uv run train_linearized_mnist.py --agent bp --optimizer sgd --lr 0.002 --num_tasks 100 --wandb --seed $seed --study_name $STUDY_NAME --weight_decay 0.0 

    uv run train_linearized_mnist.py --agent bp --optimizer sgd --lr 0.002 --num_tasks 100 --wandb --seed $seed --study_name $STUDY_NAME --weight_decay 0.005

    uv run train_linearized_mnist.py --agent bp --optimizer sgd --lr 0.002 --num_tasks 100 --wandb --seed $seed --study_name $STUDY_NAME --weight_decay 0.0 --lowrank_rank 10

    uv run train_linearized_mnist.py --agent bp --optimizer sgd --lr 0.002  --num_tasks 100 --wandb --seed $seed --study_name $STUDY_NAME --weight_decay 0.005 --lowrank_rank 10

done

echo "Done! To plot results, run:"
echo "  uv run plot_linearized_seeds.py --study_name ${STUDY_NAME}"
