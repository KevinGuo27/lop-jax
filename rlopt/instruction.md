# RL (Reinforcement Learning) Experiment Instructions

This document provides step-by-step instructions for running RL continual learning experiments, from job creation to result analysis.

## Overview

The RL workflow consists of four main steps:
1. **Job Creation**: Generate SLURM job files from hyperparameter configurations
2. **Job Submission**: Submit jobs to the cluster using SLURM
3. **Result Parsing**: Parse and aggregate experiment results
4. **Analysis & Plotting**: Generate plots and analyze performance metrics

## Prerequisites

- Access to the cluster with SLURM scheduler
- Virtual environment activated (`pobax_baseline`)
- Required Python packages installed (JAX, PPO, MuJoCo)
- Proper directory structure with hyperparameter files

## Step-by-Step Instructions

### Step 1: Generate Job Files

Navigate to the scripts directory and run the job generation script:

```bash
cd /users/kguo32/rl-opt/rlopt/scripts
python write_jobs.py
```

**What this does:**
- Reads hyperparameter configuration files from `hyperparams/nonstationary/` directory
- Generates job files in `runs/` directory (e.g., `runs_bp_hessian.txt`)
- Each line in the job file represents one experiment to run

**Available hyperparameter configurations:**
- `hyperparams/nonstationary/hessian/` - Hessian computation experiments
- `hyperparams/nonstationary/` - Standard RL experiments (bp, cbp, cbp_l2, er, l2, l2_er)

**Example job file content:**
```
python -m rlopt.ppo_nonstationary --env slippery_ant --total_steps 10000000 --num_envs 1 --num_minibatches 128 --update_epochs 10 --num_steps 2048 --vf_coeff 1.0 --weight_decay 0.0 --change_every 2000000 --compute_hessian_init --compute_hessian_end --lr 0.0001 --lambda0 0.95 --seed 2025 --n_seeds 5 --platform gpu --steps_log_freq 4 --update_log_freq 8 --debug --study_name bp_hessian
```

### Step 2: Submit Jobs to SLURM

Use the multiple SLURM jobs script to submit all experiments:

```bash
cd /users/kguo32/rl-opt/rlopt/scripts/launching
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
cd /users/kguo32/rl-opt/rlopt/scripts
python parse_experiment.py ../results
```

**What this does:**
- Scans the results directory for completed experiments
- Extracts key metrics (returns, hyperparameters)
- Aggregates results across seeds and environments
- Creates summary files (`best_hyperparam_per_env_res.pkl`)

**Output structure:**
```
results/
├── bp/
│   └── best_hyperparam_per_env_res.pkl
├── bp_hessian/
│   ├── best_hyperparam_per_env_res.pkl
│   └── slippery_ant_seed(2025)_time(...)/
├── cbp_l2/
│   └── best_hyperparam_per_env_res.pkl
├── cbp_l2_hessian/
│   ├── best_hyperparam_per_env_res.pkl
│   └── slippery_ant_seed(2025)_time(...)/
└── ...
```

### Step 4: Generate Analysis Plots

Create various analysis plots using the plotting script:

```bash
cd /users/kguo32/rl-opt/rlopt/scripts
python plot_single_metric.py
```

**Available plot types:**
- **Performance plots**: Environment-wise return evolution
- **Dead neurons**: Neuron death analysis across environments
- **Effective rank**: Model capacity analysis
- **Hessian variance**: Hessian spectrum variance analysis

**Plot outputs:**
- `slippery_ant_per_env.pdf`
- `slippery_ant_dead_neurons_per_env.pdf`
- `slippery_ant_effective_rank_per_env.pdf`

## Advanced Usage

### Custom Hyperparameter Configurations

Create new hyperparameter files in `hyperparams/nonstationary/` directory:

```python
# Example: hyperparams/nonstationary/custom_experiment.py
from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-5, 1e-4, 1e-3]  # Multiple learning rates
lambda0s = [0.9, 0.95, 0.99]  # Multiple GAE lambda values
vf_coeffs = [0.5, 1.0, 2.0]  # Multiple value function coefficients

hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': '-m rlopt.ppo_nonstationary',
    'args': [
        {
            'env': 'slippery_ant',
            'total_steps': 10000000,
            'num_envs': 1,
            'num_minibatches': 128,
            'update_epochs': 10,
            'num_steps': 2048,
            'vf_coeff': vf_coeffs,
            'weight_decay': 0.0,
            'change_every': 2000000,
            'compute_hessian_init': True,
            'compute_hessian_end': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'seed': 2025,
            'n_seeds': 5,
            'platform': 'gpu',
            'steps_log_freq': 4,
            'update_log_freq': 8,
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
input_file="../runs/runs_bp_hessian.txt"
```

### Custom Analysis

Use the hessian analysis tools for deeper investigation:

```bash
# From the main hessian_analysis directory
cd /users/kguo32/rl-opt/hessian_analysis

# Plot epsilon rank vs performance (if accuracy data available)
python plot_epsilon_rank.py --data-root /users/kguo32/rl-opt/rlopt/hessian/data --mode train --phase init

# Plot hessian variance
python plot_hessian_variance.py --data-root /users/kguo32/rl-opt/rlopt/hessian/data --mode train --phase init
```

## Dataset-Specific Notes

### RL Environment Characteristics
- **Slippery Ant**: MuJoCo-based locomotion task with non-stationary dynamics
- **PPO Algorithm**: Proximal Policy Optimization for policy learning
- **Non-stationary**: Environment changes every 2M steps
- **Long training**: 10M total steps for comprehensive evaluation
- **Hessian computation**: Computed at initialization and end of each environment

### RL-Specific Parameters
- **Total steps**: 10,000,000 steps per experiment
- **Environment changes**: Every 2,000,000 steps
- **PPO parameters**: 128 minibatches, 10 update epochs, 2048 steps per update
- **GAE lambda**: 0.95 for advantage estimation
- **Value function coefficient**: 1.0 for value loss weighting
- **Learning rate**: 1e-4 for policy and value networks

### Resource Requirements
- **Memory**: 32GB RAM sufficient
- **GPU**: Single GPU adequate for PPO training
- **Time**: 12-72 hours depending on configuration
- **Storage**: Large result files due to long training runs

## Directory Structure

```
rlopt/
├── scripts/
│   ├── hyperparams/           # Hyperparameter configurations
│   │   └── nonstationary/     # Non-stationary RL experiments
│   │       ├── hessian/       # Hessian computation experiments
│   │       └── *.py           # Standard RL experiments
│   ├── launching/
│   │   └── multiple_slurm_jobs.sh  # SLURM submission script
│   ├── runs/                  # Generated job files
│   ├── write_jobs.py          # Job generation script
│   ├── parse_experiment.py    # Result parsing script
│   ├── plot_single_metric.py  # Plotting script
│   └── plot_best_hyperparam.py # Best hyperparameter plotting
├── results/                   # Experiment results
│   ├── bp/
│   ├── bp_hessian/
│   ├── cbp_l2/
│   ├── cbp_l2_hessian/
│   └── ...
├── hessian/                   # Hessian computation results
│   ├── data/                  # Raw hessian data
│   └── *_hessian/             # Agent-specific hessian plots
├── agents/                    # RL agent implementations
├── envs/                      # Environment wrappers
├── ppo_nonstationary.py       # Main non-stationary PPO training
├── ppo_stationary.py          # Stationary PPO training
└── slurm_run.sh              # Direct SLURM execution script
```

## Troubleshooting

### Common Issues

1. **Job failures**: Check SLURM output files for error messages
2. **MuJoCo issues**: Ensure MuJoCo is properly installed and licensed
3. **Memory issues**: Monitor GPU memory usage during training
4. **Long training times**: RL experiments can take 12-72 hours
5. **Missing results**: Ensure all jobs completed successfully
6. **Parse errors**: Verify result directory structure and file formats

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

# Check training progress
tail -f results/*/slippery_ant_seed*/*.log
```

### Resource Management

- **GPU memory**: Monitor GPU usage during PPO training
- **Disk space**: Results can be large due to long training runs
- **Job limits**: Check cluster job submission limits
- **Time limits**: Adjust SLURM time limits based on experiment complexity

### RL-Specific Issues

- **Environment setup**: Ensure MuJoCo environments are properly configured
- **Policy convergence**: Monitor training curves for convergence issues
- **Non-stationarity**: Verify environment changes are occurring as expected
- **Hessian computation**: Check that hessian computation completes successfully

## Output Files

### Training Results
- Individual experiment directories with checkpoints and logs
- Aggregated result files (`best_hyperparam_per_env_res.pkl`)
- Training curves and performance metrics

### Analysis Plots
- Performance evolution plots
- Dead neuron analysis
- Effective rank analysis
- Hessian spectrum analysis

### Hessian Data
- Raw hessian spectrum data (`.npy` files)
- Hessian analysis plots (PDF files)

## Expected Timeline

- **Job generation**: < 1 minute
- **Job submission**: < 1 minute
- **Training**: 12-72 hours (depending on configuration)
- **Result parsing**: 1-5 minutes
- **Plot generation**: < 1 minute

## RL-Specific Considerations

### Environment Dynamics
- **Non-stationary**: Environment changes every 2M steps
- **Slippery Ant**: Locomotion task with varying friction coefficients
- **Long episodes**: Each environment runs for 2M steps

### Algorithm Parameters
- **PPO**: Proximal Policy Optimization with clipping
- **GAE**: Generalized Advantage Estimation for value estimation
- **Multi-seed**: 5 seeds for statistical robustness
- **Logging**: Frequent logging for monitoring training progress

### Performance Metrics
- **Returns**: Episode returns for performance evaluation
- **Dead neurons**: Neuron death analysis across environments
- **Effective rank**: Model capacity analysis
- **Hessian analysis**: Spectrum analysis for understanding optimization landscape

This workflow provides a complete pipeline for running and analyzing RL continual learning experiments with comprehensive hessian analysis capabilities.
