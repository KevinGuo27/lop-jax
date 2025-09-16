# Hessian Analysis Tools

This directory contains tools for analyzing and visualizing Hessian spectra from continual learning experiments. The tools support multiple datasets including ImageNet, Permuted MNIST, and Incremental CIFAR.

## Available Scripts

### 1. `plot_epsilon_rank.py`

Plots epsilon Hessian rank vs accuracy for different learning algorithms across tasks.

**What it does:**
- Calculates the epsilon Hessian rank (number of eigenvalues outside [-ε, +ε])
- Correlates this with task accuracy
- Supports multiple datasets and algorithms
- Handles multi-seed experiments

**Usage:**
```bash
# Basic usage with auto-detection
python plot_epsilon_rank.py --mode train --phase init --epsilon 1e-1

# Specify dataset explicitly
python plot_epsilon_rank.py --dataset incremental_cifar --mode train --phase init

# Custom paths and agents
python plot_epsilon_rank.py \
    --data-root /path/to/hessian/data \
    --results-root /path/to/results \
    --agents bp cbp l2 \
    --out-dir /path/to/output
```

**Arguments:**
- `--dataset`: Dataset type (`imagenet`, `permuted_mnist`, `incremental_cifar`)
- `--data-root`: Root directory containing hessian data
- `--results-root`: Root directory containing accuracy results
- `--agents`: List of agents to plot (default: dataset-specific)
- `--mode`: Data mode (`train` or `test`)
- `--phase`: Training phase (`init` or `end`)
- `--epsilon`: Epsilon threshold for rank calculation (default: 1e-1)
- `--out-dir`: Output directory for plots
- `--title`: Custom plot title

**Output:** PDF file showing epsilon Hessian rank vs accuracy scatter plot

### 2. `plot_hessian_variance.py`

Plots Hessian variance across tasks for different learning algorithms.

**What it does:**
- Calculates weighted variance of Hessian eigenvalues
- Shows variance evolution across tasks
- Supports eigenvalue range trimming
- Handles multi-seed experiments with error bars

**Usage:**
```bash
# Basic usage
python plot_hessian_variance.py --mode train --phase init

# With eigenvalue trimming
python plot_hessian_variance.py \
    --x-min -1000 \
    --x-max 10000 \
    --mode train \
    --phase init

# Using percentile-based trimming
python plot_hessian_variance.py \
    --x-q-lower 5 \
    --x-q-upper 95 \
    --mode train \
    --phase init
```

**Arguments:**
- `--data-root`: Root directory containing hessian data
- `--agents`: List of agents to plot
- `--mode`: Data mode (`train` or `test`)
- `--phase`: Training phase (`init` or `end`)
- `--x-min`, `--x-max`: Absolute eigenvalue cutoffs
- `--x-q-lower`, `--x-q-upper`: Percentile-based eigenvalue cutoffs
- `--out-dir`: Output directory for plots
- `--title`: Custom plot title

**Output:** PDF file showing task vs Hessian variance with mean ± SEM

### 3. `compare_hessian_plots.py`

Creates a grid comparison of Hessian spectrum plots across algorithms and tasks.

**What it does:**
- Collects existing Hessian spectrum plots (PNG/PDF files)
- Creates a grid where rows = algorithms, columns = tasks
- Useful for visual comparison of spectrum evolution

**Usage:**
```bash
python compare_hessian_plots.py \
    --hessian_dir /path/to/hessian/plots \
    --tasks 0,5,10,20,100,150,200,300,400,500,n \
    --stage at_init \
    --outfile comparison.png
```

**Arguments:**
- `--hessian_dir`: Directory containing algorithm subdirectories
- `--tasks`: Comma-separated list of task numbers (use 'n' for last task)
- `--stage`: Plot stage (`at_init` or `end`)
- `--outfile`: Output filename for comparison plot

**Output:** Grid image comparing Hessian spectra across algorithms and tasks

## Dataset Support

### ImageNet
- **Default agents:** bp, cbp, l2, l2_er, er
- **Data path:** `/users/kguo32/rl-opt/imagenet/hessian/data`
- **Results path:** `/users/kguo32/rl-opt/imagenet/results`

### Permuted MNIST
- **Default agents:** bp, cbp, l2, l2_er, er
- **Data path:** `/users/kguo32/rl-opt/permuted_mnist/hessian/data`
- **Results path:** `/users/kguo32/rl-opt/permuted_mnist/results`
- **Agent mapping:** Uses `*_hessian_fix_lr` result directories

### Incremental CIFAR
- **Default agents:** bp, cbp, l2, l2_er, er
- **Data path:** `/users/kguo32/rl-opt/incremental_cifar/hessian/data`
- **Results path:** `/users/kguo32/rl-opt/incremental_cifar/results`

## Data Format

### Hessian Data Files
- **Format:** `.npy` files containing dictionaries
- **Naming:** `hessian_task_{task_number}_{init|end}.npy`
- **Structure:** 
  ```python
  {
      "grids_train": np.array,    # Eigenvalue grid points
      "density_train": np.array,  # Corresponding densities
      "grids_test": np.array,     # Test set grids
      "density_test": np.array    # Test set densities
  }
  ```

### Results Data Files
- **Format:** `.pkl` files containing experiment results
- **Location:** `{results_root}/{agent}/best_hyperparam_per_env_res.pkl`
- **Structure:** Contains accuracy scores across seeds, hyperparameters, and tasks

## Common Issues and Solutions

### Task Index Mismatches
Some agents may have different numbers of tasks across seeds. The scripts handle this by:
- Using the first seed as reference
- Raising an error if mismatches are found
- For incremental CIFAR, problematic agents (l2_er, er) are excluded by default

### Missing Data
- Check that hessian data files exist in the expected format
- Verify results files are present and contain accuracy data
- Ensure directory structure matches expected patterns

### Large Result Files
- Some result files may be very large (>200MB)
- The scripts handle this by loading only necessary data
- Consider using smaller subsets for testing

## Examples

### Generate Epsilon Rank Plot for Incremental CIFAR
```bash
python plot_epsilon_rank.py \
    --dataset incremental_cifar \
    --mode train \
    --phase init \
    --epsilon 1e-1
```

### Generate Hessian Variance Plot with Trimming
```bash
python plot_hessian_variance.py \
    --data-root /users/kguo32/rl-opt/incremental_cifar/hessian/data \
    --agents bp cbp l2 \
    --mode train \
    --phase init \
    --x-min -1000 \
    --x-max 10000
```

### Compare Hessian Spectra
```bash
python compare_hessian_plots.py \
    --hessian_dir /users/kguo32/rl-opt/incremental_cifar/hessian \
    --tasks 0,1,2,3,4,5,6,7,8,9,10,n \
    --stage at_init \
    --outfile cifar_hessian_comparison.pdf
```

## Output Locations

- **Epsilon rank plots:** `{dataset}/hessian/plots/`
- **Variance plots:** `{dataset}/hessian/plots/`
- **Comparison plots:** Specified by `--outfile` argument

All plots are saved as PDF files (except comparison plots which are PNG) with descriptive filenames indicating the dataset, mode, and phase.
