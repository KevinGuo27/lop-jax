import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Get Paired color palette
paired_colors = cm.Paired(np.linspace(0, 1, 12))

AGENT_COLOR = {
    "l2_er": paired_colors[1],
    "er": paired_colors[3],
    "bp": paired_colors[5],
    "l2": paired_colors[7],
    "cbp": paired_colors[9],
}

AGENT_LABELS = {
    "bp": "BP",
    "cbp": "CBP", 
    "l2": "L2",
    "l2_er": "L2-ER",
    "er": "ER"
}


FNAME_RE = re.compile(r"hessian_task_(\d+)_(init|end)\.npy$")

def smooth_epsilon_ranks(epsilon_ranks, sigma=1.0):
    """Apply Gaussian smoothing to epsilon ranks to reduce scatter."""
    if len(epsilon_ranks) < 3:
        return epsilon_ranks
    return gaussian_filter1d(epsilon_ranks, sigma=sigma)

# Dataset configurations
DATASET_CONFIGS = {
    "imagenet": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er"],
        "default_data_root": Path("/users/kguo32/rl-opt/imagenet/hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/imagenet/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/imagenet/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian",
            "cbp": "cbp_hessian", 
            "l2": "l2_hessian",
            "l2_er": "l2_er_hessian",
            "er": "er_hessian"
        },
        "dataset_name": "Continual ImageNet",
        "optimal_epsilon": 8e-02
    },
    "permuted_mnist": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er"],
        "default_data_root": Path("/users/kguo32/rl-opt/permuted_mnist/hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/permuted_mnist/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/permuted_mnist/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian_fix_lr",
            "cbp": "cbp_hessian_fix_lr", 
            "l2": "l2_hessian_fix_lr",
            "l2_er": "l2_er_hessian_fix_lr",
            "er": "er_hessian_fix_lr"
        },
        "dataset_name": "Permuted MNIST",
        "optimal_epsilon": 2e-02
    },
    "incremental_cifar": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er"],
        "default_data_root": Path("/users/kguo32/rl-opt/incremental_cifar/full_hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/incremental_cifar/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/incremental_cifar/full_hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian",
            "cbp": "cbp_hessian", 
            "l2": "l2_hessian",
            "l2_er": "l2_er_hessian",
            "er": "er_hessian"
        },
        "dataset_name": "Incremental CIFAR",
        "optimal_epsilon": 4e-02
    }
}

def load_spectrum(path, mode):
    """Load hessian spectrum data from .npy file"""
    d = np.load(path, allow_pickle=True).item()
    return (d["grids_train"], d["density_train"]) if mode == "train" else (d["grids_test"], d["density_test"])

def calculate_epsilon_hessian_rank(grids, density, eps=1e-1):
    """Calculate epsilon hessian rank: number of eigenvalues outside [-eps, +eps]"""
    mask = np.abs(grids) > eps
    return np.trapz(density[mask], grids[mask])

def collect_epsilon_rank_series(seed_dir: Path, mode: str, phase: str, eps: float):
    """Return (tasks_sorted, epsilon_ranks_sorted) for one seed directory."""
    files = list(seed_dir.glob("hessian_task_*_*.npy"))
    by_task = {}
    for p in files:
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        task = int(m.group(1))
        file_phase = m.group(2)
        if file_phase != phase:
            continue
        grids, density = load_spectrum(p, mode)
        epsilon_rank = calculate_epsilon_hessian_rank(grids, density, eps)
        by_task[task] = epsilon_rank
    
    tasks = np.array(sorted(by_task.keys()))
    epsilon_ranks = np.array([by_task[t] for t in tasks])
    
    # Remove task 0 (initialization before learning)
    if len(tasks) > 0 and tasks[0] == 0:
        tasks = tasks[1:]
        epsilon_ranks = epsilon_ranks[1:]
    
    return tasks, epsilon_ranks

def gather_agent_seed_matrix(agent_dir: Path, mode: str, phase: str, eps: float):
    """Gather epsilon rank data across all seeds for an agent"""
    seed_dirs = sorted([p for p in agent_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                       key=lambda p: int(p.name))
    if len(seed_dirs) == 0:
        raise FileNotFoundError(f"No seed subdirectories found in {agent_dir}")
    
    tasks_ref, r0 = collect_epsilon_rank_series(seed_dirs[0], mode, phase, eps)
    curves = [r0]
    for sd in seed_dirs[1:]:
        t, r = collect_epsilon_rank_series(sd, mode, phase, eps)
        if not np.array_equal(t, tasks_ref):
            raise ValueError(f"Task indices mismatch across seeds in {agent_dir}: {tasks_ref} vs {t}")
        curves.append(r)
    return tasks_ref, np.stack(curves, axis=0)

def load_accuracy_data(results_dir: Path):
    """Load accuracy data from best_hyperparam_per_env_res.pkl"""
    pkl_path = results_dir / "best_hyperparam_per_env_res.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Accuracy data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    scores = data['scores']
    if scores.ndim >= 3:
        task_accuracy = np.mean(scores, axis=(0, 1)).flatten()
    else:
        task_accuracy = scores.flatten()
    
    return task_accuracy

def find_optimal_epsilon(data_root, results_root, agents, mode, phase, agent_results_map, 
                        dataset_type, epsilon_range=(1e-4, 1e-1), num_epsilons=20):
    """Find the epsilon value that gives the highest R² (best linear fit)."""
    print(f"Searching for optimal epsilon in range {epsilon_range} with {num_epsilons} values...")
    
    epsilons = np.logspace(np.log10(epsilon_range[0]), np.log10(epsilon_range[1]), num_epsilons)
    best_r_squared = 0.0
    optimal_epsilon = epsilons[0]
    
    # Load RESET baseline for incremental CIFAR if needed
    reset_accuracy = None
    if dataset_type == "incremental_cifar":
        reset_results_dir = results_root / "reset_hessian"
        if not reset_results_dir.exists():
            reset_results_dir = results_root / "reset"
        if reset_results_dir.exists():
            reset_accuracy = load_accuracy_data(reset_results_dir)
    
    for eps in epsilons:
        all_epsilon_ranks = []
        all_accuracies = []
        
        for agent in agents:
            agent_dir = data_root / agent
            tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, mode, phase, eps)
            
            results_agent = agent_results_map.get(agent, agent)
            results_dir = results_root / results_agent
            accuracy = load_accuracy_data(results_dir)
            
            # Match task indices with accuracy
            if len(accuracy) >= len(tasks):
                task_accuracy = accuracy[tasks]
            else:
                task_accuracy = np.full(len(tasks), np.nan)
                task_accuracy[:len(accuracy)] = accuracy
            
            # For incremental CIFAR, calculate difference from RESET baseline
            if dataset_type == "incremental_cifar" and reset_accuracy is not None:
                if len(reset_accuracy) >= len(tasks):
                    reset_task_accuracy = reset_accuracy[tasks]
                else:
                    reset_task_accuracy = np.full(len(tasks), np.nan)
                    reset_task_accuracy[:len(reset_accuracy)] = reset_accuracy
                task_accuracy = task_accuracy - reset_task_accuracy
            
            # Collect data points for this agent
            if dataset_type == "incremental_cifar":
                for seed_idx in range(seed_matrix.shape[0]):
                    seed_epsilon_ranks = seed_matrix[seed_idx]
                    valid_mask = ~np.isnan(task_accuracy)
                    if np.any(valid_mask):
                        all_epsilon_ranks.extend(seed_epsilon_ranks[valid_mask])
                        all_accuracies.extend(task_accuracy[valid_mask])
            else:
                avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
                valid_mask = ~np.isnan(task_accuracy)
                if np.any(valid_mask):
                    all_epsilon_ranks.extend(avg_epsilon_ranks[valid_mask])
                    all_accuracies.extend(task_accuracy[valid_mask])
        
        # Calculate R² for this epsilon
        if len(all_epsilon_ranks) > 1:
            all_epsilon_ranks = np.array(all_epsilon_ranks)
            all_accuracies = np.array(all_accuracies)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_epsilon_ranks, all_accuracies)
            r_squared = r_value ** 2
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                optimal_epsilon = eps
    
    print(f"Optimal epsilon: {optimal_epsilon:.2e} with R² = {best_r_squared:.4f}")
    return optimal_epsilon, best_r_squared

def plot_dataset_subplot(ax, dataset_type, mode="train", phase="init", epsilon=None, smooth_sigma=1.0):
    """Plot epsilon hessian rank vs accuracy for a single dataset on given axis"""
    
    # Get dataset configuration
    config = DATASET_CONFIGS[dataset_type]
    data_root = config["default_data_root"]
    results_root = config["default_results_root"]
    agents = config["default_agents"]
    agent_results_map = config["agent_results_map"]
    dataset_name = config["dataset_name"]
    
    # Use optimal epsilon from configuration or search if not available
    if epsilon is None:
        if "optimal_epsilon" in config:
            # Use pre-determined optimal epsilon
            epsilon = config["optimal_epsilon"]
        else:
            # Search for optimal epsilon
            epsilon_range = (1e-2, 1.0)
            epsilon, _ = find_optimal_epsilon(
                data_root, results_root, agents, mode, phase, 
                agent_results_map, dataset_type, epsilon_range, 50
            )
    
    print(f"Using epsilon for {dataset_name}: {epsilon:.2e}")
    
    # Collect all data points for linear regression
    all_epsilon_ranks = []
    all_accuracies = []
    
    # For incremental CIFAR, load RESET baseline accuracy for comparison
    reset_accuracy = None
    if dataset_type == "incremental_cifar":
        reset_results_dir = results_root / "reset_hessian"
        if not reset_results_dir.exists():
            reset_results_dir = results_root / "reset"
        if reset_results_dir.exists():
            reset_accuracy = load_accuracy_data(reset_results_dir)
    
    for agent in agents:
        agent_dir = data_root / agent
        color_hex = AGENT_COLOR.get(agent, "black")
        
        tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, mode, phase, epsilon)
        
        # For incremental CIFAR, limit to first 10 tasks
        if dataset_type == "incremental_cifar":
            max_tasks = min(10, len(tasks))
            tasks = tasks[:max_tasks]
            seed_matrix = seed_matrix[:, :max_tasks]
        
        results_agent = agent_results_map.get(agent, agent)
        results_dir = results_root / results_agent
        accuracy = load_accuracy_data(results_dir)
        
        # Match task indices with accuracy
        if len(accuracy) >= len(tasks):
            task_accuracy = accuracy[tasks]
        else:
            task_accuracy = np.full(len(tasks), np.nan)
            task_accuracy[:len(accuracy)] = accuracy
        
        # For incremental CIFAR, limit to first 10 tasks
        if dataset_type == "incremental_cifar":
            max_tasks = min(10, len(tasks))
            tasks = tasks[:max_tasks]
            task_accuracy = task_accuracy[:max_tasks]
        
        # For incremental CIFAR, calculate difference from RESET baseline
        if dataset_type == "incremental_cifar" and reset_accuracy is not None:
            if len(reset_accuracy) >= len(tasks):
                reset_task_accuracy = reset_accuracy[tasks]
            else:
                reset_task_accuracy = np.full(len(tasks), np.nan)
                reset_task_accuracy[:len(reset_accuracy)] = reset_accuracy
            task_accuracy = task_accuracy - reset_task_accuracy
        
        # Plot data points
        if dataset_type == "incremental_cifar":
            alpha_value = 0.5 if agent != "l2_er" else 1.0
            for seed_idx in range(seed_matrix.shape[0]):
                seed_epsilon_ranks = seed_matrix[seed_idx]
                seed_epsilon_ranks = smooth_epsilon_ranks(seed_epsilon_ranks, sigma=smooth_sigma)
                
                valid_mask = ~np.isnan(task_accuracy)
                if np.any(valid_mask):
                    all_epsilon_ranks.extend(seed_epsilon_ranks[valid_mask])
                    all_accuracies.extend(task_accuracy[valid_mask])
                
                label = AGENT_LABELS.get(agent) if seed_idx == 0 else None
                ax.scatter(seed_epsilon_ranks, task_accuracy, 
                         s=20, alpha=alpha_value, label=label, color=color_hex)
        else:
            avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
            valid_mask = ~np.isnan(task_accuracy)
            if np.any(valid_mask):
                all_epsilon_ranks.extend(avg_epsilon_ranks[valid_mask])
                all_accuracies.extend(task_accuracy[valid_mask])
            
            alpha_value = 0.5 if agent != "l2_er" else 1.0
            ax.scatter(avg_epsilon_ranks, task_accuracy, 
                     s=20, alpha=alpha_value, label=AGENT_LABELS.get(agent), color=color_hex)
    
    # Calculate and plot linear regression line
    r_squared = None
    if len(all_epsilon_ranks) > 1:
        all_epsilon_ranks = np.array(all_epsilon_ranks)
        all_accuracies = np.array(all_accuracies)
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_epsilon_ranks, all_accuracies)
        r_squared = r_value**2
        
        # Create line for plotting
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, 'k--', alpha=0.8, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel(f"Percentage of eigenvalues outside $(-{float(epsilon):.2f}, {float(epsilon):.2f})$", fontsize=32)
    
    # Set y-axis label based on dataset type
    if dataset_type == "incremental_cifar" and reset_accuracy is not None:
        ax.set_ylabel("Accuracy Difference from RESET", fontsize=32)
    else:
        ax.set_ylabel("Accuracy", fontsize=32)
    
    # Add R² value to the title if available
    if r_squared is not None:
        ax.set_title(f"{dataset_name} ($R^2$={r_squared:.3f})", fontsize=32)
    else:
        ax.set_title(dataset_name, fontsize=32)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # Fix minus sign display on y-axis for incremental CIFAR
    if dataset_type == "incremental_cifar" and reset_accuracy is not None:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    return ax

def plot_dataset_subplot_full(ax, dataset_type, mode="train", phase="init", epsilon=None, smooth_sigma=1.0):
    """Plot epsilon hessian rank vs accuracy for a single dataset on given axis (full data, no task limit)"""
    
    # Get dataset configuration
    config = DATASET_CONFIGS[dataset_type]
    data_root = config["default_data_root"]
    results_root = config["default_results_root"]
    agents = config["default_agents"]
    agent_results_map = config["agent_results_map"]
    dataset_name = config["dataset_name"]
    
    # Use optimal epsilon from configuration or search if not available
    if epsilon is None:
        if "optimal_epsilon" in config:
            # Use pre-determined optimal epsilon
            epsilon = config["optimal_epsilon"]
        else:
            # Search for optimal epsilon
            epsilon_range = (1e-2, 1.0)
            epsilon, _ = find_optimal_epsilon(
                data_root, results_root, agents, mode, phase, 
                agent_results_map, dataset_type, epsilon_range, 50
            )
    
    print(f"Using epsilon for {dataset_name} (full): {epsilon:.2e}")
    
    # Collect all data points for linear regression
    all_epsilon_ranks = []
    all_accuracies = []
    
    # For incremental CIFAR, load RESET baseline accuracy for comparison
    reset_accuracy = None
    if dataset_type == "incremental_cifar":
        reset_results_dir = results_root / "reset_hessian"
        if not reset_results_dir.exists():
            reset_results_dir = results_root / "reset"
        if reset_results_dir.exists():
            reset_accuracy = load_accuracy_data(reset_results_dir)
    
    for agent in agents:
        agent_dir = data_root / agent
        color_hex = AGENT_COLOR.get(agent, "black")
        
        tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, mode, phase, epsilon)
        # NOTE: No task limiting for full plot
        
        results_agent = agent_results_map.get(agent, agent)
        results_dir = results_root / results_agent
        accuracy = load_accuracy_data(results_dir)
        
        # Match task indices with accuracy
        if len(accuracy) >= len(tasks):
            task_accuracy = accuracy[tasks]
        else:
            task_accuracy = np.full(len(tasks), np.nan)
            task_accuracy[:len(accuracy)] = accuracy
        
        # For incremental CIFAR, calculate difference from RESET baseline
        if dataset_type == "incremental_cifar" and reset_accuracy is not None:
            if len(reset_accuracy) >= len(tasks):
                reset_task_accuracy = reset_accuracy[tasks]
            else:
                reset_task_accuracy = np.full(len(tasks), np.nan)
                reset_task_accuracy[:len(reset_accuracy)] = reset_accuracy
            task_accuracy = task_accuracy - reset_task_accuracy
        
        # Plot data points
        if dataset_type == "incremental_cifar":
            alpha_value = 0.5 if agent != "l2_er" else 1.0
            for seed_idx in range(seed_matrix.shape[0]):
                seed_epsilon_ranks = seed_matrix[seed_idx]
                seed_epsilon_ranks = smooth_epsilon_ranks(seed_epsilon_ranks, sigma=smooth_sigma)
                
                valid_mask = ~np.isnan(task_accuracy)
                if np.any(valid_mask):
                    all_epsilon_ranks.extend(seed_epsilon_ranks[valid_mask])
                    all_accuracies.extend(task_accuracy[valid_mask])
                
                label = AGENT_LABELS.get(agent) if seed_idx == 0 else None
                ax.scatter(seed_epsilon_ranks, task_accuracy, 
                         s=20, alpha=alpha_value, label=label, color=color_hex)
        else:
            avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
            valid_mask = ~np.isnan(task_accuracy)
            if np.any(valid_mask):
                all_epsilon_ranks.extend(avg_epsilon_ranks[valid_mask])
                all_accuracies.extend(task_accuracy[valid_mask])
            
            alpha_value = 0.5 if agent != "l2_er" else 1.0
            ax.scatter(avg_epsilon_ranks, task_accuracy, 
                     s=20, alpha=alpha_value, label=AGENT_LABELS.get(agent), color=color_hex)
    
    # Calculate and plot linear regression line
    r_squared = None
    if len(all_epsilon_ranks) > 1:
        all_epsilon_ranks = np.array(all_epsilon_ranks)
        all_accuracies = np.array(all_accuracies)
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_epsilon_ranks, all_accuracies)
        r_squared = r_value**2
        
        # Create line for plotting
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, 'k--', alpha=0.8, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel(f"Percentage of eigenvalues outside $(-{float(epsilon):.2f}, {float(epsilon):.2f})$", fontsize=32)
    
    # Set y-axis label based on dataset type
    if dataset_type == "incremental_cifar" and reset_accuracy is not None:
        ax.set_ylabel("Accuracy Difference from RESET", fontsize=32)
    else:
        ax.set_ylabel("Accuracy", fontsize=32)
    
    # Add R² value to the title if available
    if r_squared is not None:
        ax.set_title(f"{dataset_name} ($R^2$={r_squared:.3f})", fontsize=32)
    else:
        ax.set_title(dataset_name, fontsize=32)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # Fix minus sign display on y-axis for incremental CIFAR
    if dataset_type == "incremental_cifar" and reset_accuracy is not None:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    return ax

def main():
    # Set font to serif like in plot_single_metric.py
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']
    
    # Create figure with 2x1 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # Define datasets to plot
    datasets = ["permuted_mnist", "incremental_cifar"]
    
    # Plot each dataset
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        plot_dataset_subplot(axes[i], dataset, mode="train", phase="init")
    
    # Create a single legend for all subplots at the bottom
    # Get handles and labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Create new legend handles with full opacity
    import matplotlib.lines as mlines
    legend_handles = []
    for handle in handles:
        # Create new handle with alpha=1.0 (full opacity)
        new_handle = mlines.Line2D([], [], color=handle.get_facecolors()[0], 
                                  marker='o', linestyle='None', markersize=8, 
                                  alpha=1.0, label=handle.get_label())
        legend_handles.append(new_handle)
    
    # Create legend at the bottom of the figure
    fig.legend(legend_handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=len(legend_handles), fontsize=32)
    
    # Adjust layout to make room for the legend and add vertical spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.3)
    
    # Save the combined plot
    out_dir = Path("/users/kguo32/rl-opt/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "scatterplots.pdf"
    
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[INFO] Saved combined plot: {out_pdf}")
    plt.close()
    
    # Create individual plot for full 20-task incremental CIFAR
    print("Creating individual plot for full 20-task incremental CIFAR...")
    fig_individual, ax_individual = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot incremental CIFAR with all 20 tasks (temporarily disable the 10-task limit)
    plot_dataset_subplot_full(ax_individual, "incremental_cifar", mode="train", phase="init")
    
    # Create legend for individual plot
    handles_ind, labels_ind = ax_individual.get_legend_handles_labels()
    
    # Create new legend handles with full opacity for individual plot
    legend_handles_ind = []
    for handle in handles_ind:
        new_handle = mlines.Line2D([], [], color=handle.get_facecolors()[0], 
                                  marker='o', linestyle='None', markersize=8, 
                                  alpha=1.0, label=handle.get_label())
        legend_handles_ind.append(new_handle)
    
    # Add legend to individual plot
    ax_individual.legend(handles=legend_handles_ind, labels=labels_ind, 
                        loc='best', fontsize=24)
    
    plt.tight_layout()
    
    # Save individual plot
    out_pdf_individual = out_dir / "incremental_cifar_full_20_tasks.pdf"
    plt.savefig(out_pdf_individual, bbox_inches="tight")
    print(f"[INFO] Saved individual 20-task plot: {out_pdf_individual}")
    plt.close()

if __name__ == "__main__":
    main()
