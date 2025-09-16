# Permuted MNIST Experiment Instructions

This document provides step-by-step instructions for running Permuted MNIST continual learning experiments, from job creation to result analysis.

## Overview

The Permuted MNIST workflow consists of four main steps:
1. **Job Creation**: Generate SLURM job files from hyperparameter configurations
2. **Job Submission**: Submit jobs to the cluster using SLURM
3. **Result Parsing**: Parse and aggregate experiment results
4. **Analysis & Plotting**: Generate plots and analyze performance metrics

## Prerequisites

- Access to the cluster with SLURM scheduler
- Virtual environment activated (`lop` or `pobax_baseline`)
- Required Python packages installed
- Proper directory structure with hyperparameter files

## Step-by-Step Instructions

### Step 1: Generate Job Files

Navigate to the scripts directory and run the job generation script:

```bash
cd /users/kguo32/rl-opt/permuted_mnist/scripts
python write_jobs.py
```

**What this does:**
- Reads hyperparameter configuration files from `hyperparams/` directory
- Generates job files in `runs/` directory (e.g., `runs_bp_hessian_fix_lr.txt`)
- Each line in the job file represents one experiment to run

**Available hyperparameter configurations:**
- `hyperparams/hessian/fix_lr/` - Hessian computation with fixed learning rates
- `hyperparams/changelr/` - Different learning rate configurations
- `hyperparams/hessian/` - Standard hessian computation experiments

**Example job file content:**
```
python -m permuted_mnist.train_permuted_mnist --agent bp --weight_decay 0.0 --num_features 1000 --compute_hessian True --compute_hessian_interval 10 --lr 0.01 --seed 2025 --n_seeds 1 --platform gpu --debug True --study_name bp_hessian_fix_lr
python -m permuted_mnist.train_permuted_mnist --agent bp --weight_decay 0.0 --num_features 1000 --compute_hessian True --compute_hessian_interval 10 --lr 0.01 --seed 2026 --n_seeds 1 --platform gpu --debug True --study_name bp_hessian_fix_lr
...
```

### Step 2: Submit Jobs to SLURM

Use the multiple SLURM jobs script to submit all experiments:

```bash
cd /users/kguo32/rl-opt/permuted_mnist/scripts/launching
sbatch multiple_slurm_jobs.sh
```

**What this does:**
- Reads job files from `../runs/` directory
- Submits each line as a separate SLURM job
- Uses GPU partition with appropriate resource allocation
- Creates individual output/error files for each job

**SLURM Configuration:**
- Partition: `3090-gcondo`
- Resources: 1 GPU, 3 CPUs, 32GB RAM
- Time limit: 72 hours
- Excluded nodes: `gpu2106,gpu2102,gpu2115,gpu2105`

**Monitor job status:**
```bash
squeue -u $USER
```

### Step 3: Parse Experiment Results

After all jobs complete, parse and aggregate the results:

```bash
cd /users/kguo32/rl-opt/permuted_mnist/scripts
python parse_experiments.py ../results
```

**What this does:**
- Scans the results directory for completed experiments
- Extracts key metrics (accuracy, hyperparameters)
- Aggregates results across seeds and hyperparameters
- Creates summary files (`best_hyperparam_per_env_res.pkl`)

**Output structure:**
```
results/
├── bp_hessian_fix_lr/
│   ├── best_hyperparam_per_env_res.pkl
│   └── permuted_mnist_seed(2025)_time(...)/
├── cbp_hessian_fix_lr/
│   ├── best_hyperparam_per_env_res.pkl
│   └── permuted_mnist_seed(2025)_time(...)/
└── ...
```

### Step 4: Generate Analysis Plots

Create various analysis plots using the plotting script:

```bash
cd /users/kguo32/rl-opt/permuted_mnist/scripts
python plot_single_metric.py
```

**Available plot types:**
- **Accuracy plots**: Task-wise accuracy evolution
- **Dead neurons**: Neuron death analysis across tasks
- **Effective rank**: Model capacity analysis
- **Hessian variance**: Hessian spectrum variance analysis

**Plot outputs:**
- `permuted_mnist_accuracy_eval_per_task.pdf`
- `permuted_mnist_dead_neurons_per_task.pdf`
- `permuted_mnist_effective_rank_per_task.pdf`
- `hessian_variance_train_init_multiseed.pdf`

## Advanced Usage

### Custom Hyperparameter Configurations

Create new hyperparameter files in `hyperparams/` directory:

```python
# Example: hyperparams/custom_experiment.py
from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-3, 1e-2, 1e-1]  # Multiple learning rates
weight_decays = [0.0, 1e-4, 1e-3]  # Multiple weight decay values

hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'bp',
            'weight_decay': weight_decays,
            'num_features': 1000,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'lr': lrs,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}
```

### Selective Job Submission

Submit specific job files instead of all:

```bash
# Modify multiple_slurm_jobs.sh to use specific input file
input_file="../runs/runs_bp_hessian_fix_lr.txt"
```

### Custom Analysis

Use the hessian analysis tools for deeper investigation:

```bash
# From the main hessian_analysis directory
cd /users/kguo32/rl-opt/hessian_analysis

# Plot epsilon rank vs accuracy
python plot_epsilon_rank.py --dataset permuted_mnist --mode train --phase init

# Plot hessian variance
python plot_hessian_variance.py --data-root /users/kguo32/rl-opt/permuted_mnist/hessian/data --mode train --phase init
```

## Directory Structure

```
permuted_mnist/
├── scripts/
│   ├── hyperparams/           # Hyperparameter configurations
│   │   ├── hessian/
│   │   │   ├── fix_lr/        # Fixed LR hessian experiments
│   │   │   └── *.py           # Other hessian configs
│   │   ├── changelr/          # Learning rate experiments
│   │   └── *.py               # Standard experiments
│   ├── launching/
│   │   └── multiple_slurm_jobs.sh  # SLURM submission script
│   ├── runs/                  # Generated job files
│   ├── write_jobs.py          # Job generation script
│   ├── parse_experiments.py   # Result parsing script
│   └── plot_single_metric.py  # Plotting script
├── results/                   # Experiment results
│   ├── bp_hessian_fix_lr/
│   ├── cbp_hessian_fix_lr/
│   └── ...
├── hessian/                   # Hessian computation results
│   ├── data/                  # Raw hessian data
│   └── plots/                 # Hessian analysis plots
└── train_permuted_mnist.py    # Main training script
```

## Troubleshooting

### Common Issues

1. **Job failures**: Check SLURM output files for error messages
2. **Missing results**: Ensure all jobs completed successfully
3. **Parse errors**: Verify result directory structure and file formats
4. **Plot generation issues**: Check that parsed results exist

### Monitoring Commands

```bash
# Check job status
squeue -u $USER

# View job output
cat kevin-*.out

# Check for errors
cat kevin-*.err

# Monitor disk usage
du -sh results/

# Check result files
ls -la results/*/best_hyperparam_per_env_res.pkl
```

### Resource Management

- **GPU memory**: Monitor GPU usage during training
- **Disk space**: Results can be large, monitor available space
- **Job limits**: Check cluster job submission limits
- **Time limits**: Adjust SLURM time limits based on experiment complexity

## Output Files

### Training Results
- Individual experiment directories with checkpoints and logs
- Aggregated result files (`best_hyperparam_per_env_res.pkl`)

### Analysis Plots
- Accuracy evolution plots
- Dead neuron analysis
- Effective rank analysis
- Hessian spectrum analysis

### Hessian Data
- Raw hessian spectrum data (`.npy` files)
- Hessian analysis plots (PDF files)

This workflow provides a complete pipeline for running and analyzing Permuted MNIST continual learning experiments with comprehensive hessian analysis capabilities.
