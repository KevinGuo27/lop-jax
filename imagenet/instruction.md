# ImageNet Experiment Instructions

This document provides step-by-step instructions for running ImageNet continual learning experiments, from job creation to result analysis.

## Overview

The ImageNet workflow consists of four main steps:
1. **Job Creation**: Generate SLURM job files from hyperparameter configurations
2. **Job Submission**: Submit jobs to the cluster using SLURM
3. **Result Parsing**: Parse and aggregate experiment results
4. **Analysis & Plotting**: Generate plots and analyze performance metrics

## Prerequisites

- Access to the cluster with SLURM scheduler
- Virtual environment activated (`pobax_baseline`)
- Required Python packages installed
- Proper directory structure with hyperparameter files

## Step-by-Step Instructions

### Step 1: Generate Job Files

Navigate to the scripts directory and run the job generation script:

```bash
cd /users/kguo32/rl-opt/imagenet/scripts
python write_jobs.py
```

**What this does:**
- Reads hyperparameter configuration files from `hyperparams/` directory
- Generates job files in `runs/` directory (e.g., `runs_bp_hessian.txt`)
- Each line in the job file represents one experiment to run

**Available hyperparameter configurations:**
- `hyperparams/hessian/` - Hessian computation experiments
- `hyperparams/` - Standard experiments (bp, cbp, er, l2, l2_er, snp, snp_l2)

**Example job file content:**
```
python -m imagenet.train_imagenet --agent bp --weight_decay 0.0 --lr 0.0001 --seed 2035 --num_tasks 2000 --compute_hessian True --compute_hessian_interval 10 --n_seeds 1 --platform gpu --debug False --study_name bp_hessian
python -m imagenet.train_imagenet --agent bp --weight_decay 0.0 --lr 0.0001 --seed 2036 --num_tasks 2000 --compute_hessian True --compute_hessian_interval 10 --n_seeds 1 --platform gpu --debug False --study_name bp_hessian
...
```

### Step 2: Submit Jobs to SLURM

Use the multiple SLURM jobs script to submit all experiments:

```bash
cd /users/kguo32/rl-opt/imagenet/scripts/launching
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
- Excluded nodes: `gpu2106,gpu2102,gpu2115,gpu2105,gpu2110,gpu2603,gpu2604`

**Monitor job status:**
```bash
squeue -u $USER
```

### Step 3: Parse Experiment Results

After all jobs complete, parse and aggregate the results:

```bash
cd /users/kguo32/rl-opt/imagenet/scripts
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
├── bp/
│   └── best_hyperparam_per_env_res.pkl
├── bp_hessian/
│   ├── best_hyperparam_per_env_res.pkl
│   └── imagenet_seed(2035)_time(...)/
├── cbp/
│   └── best_hyperparam_per_env_res.pkl
├── cbp_hessian/
│   ├── best_hyperparam_per_env_res.pkl
│   └── imagenet_seed(2035)_time(...)/
└── ...
```

### Step 4: Generate Analysis Plots

Create various analysis plots using the plotting script:

```bash
cd /users/kguo32/rl-opt/imagenet/scripts
python plot_single_metric.py
```

**Available plot types:**
- **Accuracy plots**: Task-wise accuracy evolution
- **Dead neurons**: Neuron death analysis across tasks
- **Effective rank**: Model capacity analysis
- **Hessian variance**: Hessian spectrum variance analysis

**Plot outputs:**
- `imagenet_accuracy_eval_per_task.pdf`
- `imagenet_accuracy_eval_hessian_per_task.pdf`
- `imagenet_dead_neurons_per_task.pdf`
- `imagenet_effective_rank_per_task.pdf`

## Advanced Usage

### Custom Hyperparameter Configurations

Create new hyperparameter files in `hyperparams/` directory:

```python
# Example: hyperparams/custom_experiment.py
from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-5, 1e-4, 1e-3]  # Multiple learning rates
weight_decays = [0.0, 1e-4, 1e-3]  # Multiple weight decay values

hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'bp',
            'weight_decay': weight_decays,
            'lr': lrs,
            'seed': [2035 + i for i in range(10)],
            'num_tasks': 2000,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}
```

### Selective Job Submission

Submit specific job files instead of all:

```bash
# Modify multiple_slurm_jobs.sh to use specific input file
input_file="../runs/runs_bp_hessian.txt"
```

### Custom Analysis

Use the hessian analysis tools for deeper investigation:

```bash
# From the main hessian_analysis directory
cd /users/kguo32/rl-opt/hessian_analysis

# Plot epsilon rank vs accuracy
python plot_epsilon_rank.py --dataset imagenet --mode train --phase init

# Plot hessian variance
python plot_hessian_variance.py --data-root /users/kguo32/rl-opt/imagenet/hessian/data --mode train --phase init
```

## Dataset-Specific Notes

### ImageNet Characteristics
- **Large-scale dataset**: 1000 classes, 1.2M training images
- **High computational requirements**: Longer training times
- **More seeds**: Typically uses 20 seeds (2035-2054) for statistical robustness
- **Lower learning rates**: Default LR of 1e-4 for stability
- **More tasks**: 2000 tasks for comprehensive continual learning evaluation

### Resource Requirements
- **Memory**: 32GB RAM minimum
- **GPU**: Single GPU sufficient for most experiments
- **Time**: 12-72 hours depending on configuration
- **Storage**: Large result files due to dataset size

## Directory Structure

```
imagenet/
├── scripts/
│   ├── hyperparams/           # Hyperparameter configurations
│   │   ├── hessian/           # Hessian computation experiments
│   │   └── *.py               # Standard experiments
│   ├── launching/
│   │   └── multiple_slurm_jobs.sh  # SLURM submission script
│   ├── runs/                  # Generated job files
│   ├── write_jobs.py          # Job generation script
│   ├── parse_experiments.py   # Result parsing script
│   └── plot_single_metric.py  # Plotting script
├── results/                   # Experiment results
│   ├── bp/
│   ├── bp_hessian/
│   ├── cbp/
│   ├── cbp_hessian/
│   └── ...
├── hessian/                   # Hessian computation results
│   ├── data/                  # Raw hessian data
│   └── plots/                 # Hessian analysis plots
├── class_order               # ImageNet class ordering
└── train_imagenet.py         # Main training script
```

## Troubleshooting

### Common Issues

1. **Job failures**: Check SLURM output files for error messages
2. **Memory issues**: ImageNet requires significant memory, monitor usage
3. **Long training times**: ImageNet experiments can take 12-72 hours
4. **Missing results**: Ensure all jobs completed successfully
5. **Parse errors**: Verify result directory structure and file formats

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

# Monitor GPU usage
nvidia-smi
```

### Resource Management

- **GPU memory**: Monitor GPU usage during training
- **Disk space**: Results can be very large, monitor available space
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

This workflow provides a complete pipeline for running and analyzing ImageNet continual learning experiments with comprehensive hessian analysis capabilities.
