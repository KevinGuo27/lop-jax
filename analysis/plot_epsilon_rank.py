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
    "laynorm_l2": paired_colors[0],
    "spectral_reg": paired_colors[2],
}

AGENT_LABELS = {
    "bp": "BP",
    "cbp": "CBP", 
    "l2": "L2",
    "l2_er": "L2-ER",
    "er": "ER",
    "laynorm_l2": "LayerNorm-L2",
    "spectral_reg": "Spectral"
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
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er", "laynorm_l2", "spectral_reg"],
        "default_data_root": Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/imagenet/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/imagenet/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian",
            "cbp": "cbp_hessian",
            "l2": "l2_hessian",
            "l2_er": "l2_er_hessian",
            "er": "er_hessian",
            "laynorm_l2": "laynorm_l2_hessian",
            "spectral_reg": "spectral_reg_hessian",
        },
        "dataset_name": "Continual ImageNet",
        # "optimal_epsilon": 8e-02,
    },
    "permuted_mnist": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er", "laynorm_l2", "spectral_reg"],
        "default_data_root": Path("/users/kguo32/rl-opt/permuted_mnist/hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/permuted_mnist/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/permuted_mnist/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian_fix_lr",
            "cbp": "cbp_hessian_fix_lr",
            "l2": "l2_hessian_fix_lr",
            "l2_er": "l2_er_hessian_fix_lr",
            "er": "er_hessian_fix_lr",
            "laynorm_l2": "laynorm_l2_hessian_fix_lr",
            "spectral_reg": "spectral_reg_hessian_fix_lr"
        },
        "dataset_name": "Permuted MNIST",
        # "optimal_epsilon": 2e-02
    },
    "incremental_cifar": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er", "layernorm_l2", "spectral_reg"],
        "default_data_root": Path("/users/kguo32/rl-opt/incremental_cifar/hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/incremental_cifar/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/incremental_cifar/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian",
            "cbp": "cbp_hessian",
            "l2": "l2_hessian",
            "l2_er": "l2_er_hessian",
            "er": "er_hessian",
            "layernorm_l2": "layernorm_l2",
            "spectral_reg": "spec_reg_hessian",
        },
        "dataset_name": "Incremental CIFAR",
        # "optimal_epsilon": 4e-02
    }
}

def load_spectrum(path, mode):
    """Load hessian spectrum data from .npy file"""
    d = np.load(path, allow_pickle=True).item()
    return (d["grids_train"], d["density_train"]) if mode == "train" else (d["grids_test"], d["density_test"])

def calculate_epsilon_hessian_rank(grids, density, eps=1e-1):
    """Calculate epsilon hessian rank: percentage of eigenvalues outside [-eps, +eps]"""
    # Calculate the total integral (should be ~1.0 for normalized density)
    total_integral = np.trapz(density, grids)
    
    # Calculate integral outside [-eps, +eps]
    mask = np.abs(grids) > eps
    outside_integral = np.trapz(density[mask], grids[mask])
    
    # Return as percentage
    return (outside_integral / total_integral) * 100.0

def collect_epsilon_rank_series(seed_dir: Path, mode: str, phase: str, eps: float, dataset_type: str = None):
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
    
    # For incremental CIFAR, limit to first 10 tasks
    if dataset_type == "incremental_cifar" and len(tasks) > 10:
        tasks = tasks[:10]
        epsilon_ranks = epsilon_ranks[:10]
    
    return tasks, epsilon_ranks

def gather_agent_seed_matrix(agent_dir: Path, mode: str, phase: str, eps: float, dataset_type: str = None):
    """Gather epsilon rank data across all seeds for an agent"""
    seed_dirs = sorted([p for p in agent_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                       key=lambda p: int(p.name))
    if len(seed_dirs) == 0:
        raise FileNotFoundError(f"No seed subdirectories found in {agent_dir}")
    
    tasks_ref, r0 = collect_epsilon_rank_series(seed_dirs[0], mode, phase, eps, dataset_type)
    curves = [r0]
    for sd in seed_dirs[1:]:
        t, r = collect_epsilon_rank_series(sd, mode, phase, eps, dataset_type)
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

def detect_dataset_type(data_root: Path) -> str:
    """Detect dataset type based on data root path"""
    data_root_str = str(data_root).lower()
    if "permuted_mnist" in data_root_str or "mnist" in data_root_str:
        return "permuted_mnist"
    elif "incremental_cifar" in data_root_str or "cifar" in data_root_str:
        return "incremental_cifar"
    else:
        return "imagenet"

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
            tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, mode, phase, eps, dataset_type)
            # For incremental CIFAR, only use the first 10 tasks to pick epsilon
            if dataset_type == "incremental_cifar" and len(tasks) > 10:
                tasks = tasks[:10]
                seed_matrix = seed_matrix[:, :10]
            
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
                
            print(f"Epsilon {eps:.2e}: R² = {r_squared:.4f}")
    
    print(f"Optimal epsilon: {optimal_epsilon:.2e} with R² = {best_r_squared:.4f}")
    return optimal_epsilon, best_r_squared

def main():
    parser = argparse.ArgumentParser(description="Plot epsilon hessian rank vs accuracy")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "permuted_mnist", "incremental_cifar"], 
                        default=None, help="Dataset type (auto-detected if not specified)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Root directory containing hessian data")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Root directory containing results data")
    parser.add_argument("--agents", type=str, nargs="*", default=None,
                        help="List of agents to plot")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--phase", type=str, choices=["init", "end"], default="init")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Fixed epsilon threshold (if not provided, will search for optimal)")
    parser.add_argument("--epsilon-min", type=float, default=1e-2,
                        help="Minimum epsilon for search range")
    parser.add_argument("--epsilon-max", type=float, default=1.0,
                        help="Maximum epsilon for search range")
    parser.add_argument("--num-epsilons", type=int, default=100,
                        help="Number of epsilon values to test in search")
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Sigma parameter for gaussian smoothing of epsilon ranks")

    args = parser.parse_args()
    
    # Set font to serif
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']

    # Determine dataset type
    dataset_type = args.dataset or detect_dataset_type(args.data_root or Path("."))
    config = DATASET_CONFIGS[dataset_type]
    
    # Set defaults based on dataset
    data_root = args.data_root or config["default_data_root"]
    results_root = args.results_root or config["default_results_root"]
    agents = args.agents or config["default_agents"]
    out_dir = args.out_dir or config["default_out_dir"]
    agent_results_map = config["agent_results_map"]
    dataset_name = config["dataset_name"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine epsilon to use
    if args.epsilon is not None:
        epsilon_to_use = args.epsilon
        print(f"Using fixed epsilon: {epsilon_to_use}")
    elif "optimal_epsilon" in config:
        epsilon_to_use = config["optimal_epsilon"]
        print(f"Using configured optimal epsilon: {epsilon_to_use:.2e}")
    else:
        epsilon_range = (args.epsilon_min, args.epsilon_max)
        epsilon_to_use, best_r_squared = find_optimal_epsilon(
            data_root, results_root, agents, args.mode, args.phase,
            agent_results_map, dataset_type, epsilon_range, args.num_epsilons
        )
        print(f"Using optimal epsilon: {epsilon_to_use:.2e} (R² = {best_r_squared:.4f})")

    plt.figure(figsize=(10, 6))

    # Collect all data points for linear regression
    all_epsilon_ranks = []
    all_accuracies = []

    # Load RESET baseline for incremental CIFAR if needed
    reset_accuracy = None
    if dataset_type == "incremental_cifar":
        reset_results_dir = results_root / "reset_hessian"
        if not reset_results_dir.exists():
            reset_results_dir = results_root / "reset"
        if reset_results_dir.exists():
            reset_accuracy = load_accuracy_data(reset_results_dir)
            print(f"Loaded RESET baseline accuracy from: {reset_results_dir}")

    for agent in agents:
        agent_dir = data_root / agent
        color_hex = AGENT_COLOR.get(agent, "black")
        
        print(f"Processing agent: {agent}")
        
        tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, args.mode, args.phase, epsilon_to_use, dataset_type)
        
        results_agent = agent_results_map.get(agent, agent)
        results_dir = results_root / results_agent
        accuracy = load_accuracy_data(results_dir)
        
        # Match task indices with accuracy
        if len(accuracy) >= len(tasks):
            task_accuracy = accuracy[tasks]
        else:
            task_accuracy = np.full(len(tasks), np.nan)
            task_accuracy[:len(accuracy)] = accuracy
        
        # For incremental CIFAR, also limit accuracy to first 10 tasks  
        if dataset_type == "incremental_cifar" and len(task_accuracy) > 10:
            task_accuracy = task_accuracy[:10]
        
        # For incremental CIFAR, calculate difference from RESET baseline
        if dataset_type == "incremental_cifar" and reset_accuracy is not None:
            if len(reset_accuracy) >= len(tasks):
                reset_task_accuracy = reset_accuracy[tasks]
            else:
                reset_task_accuracy = np.full(len(tasks), np.nan)
                reset_task_accuracy[:len(reset_accuracy)] = reset_accuracy
            
            # Also limit RESET baseline to first 10 tasks
            if len(reset_task_accuracy) > 10:
                reset_task_accuracy = reset_task_accuracy[:10]
                
            task_accuracy = task_accuracy - reset_task_accuracy
        
        # Plot data points
        if dataset_type == "incremental_cifar":
            alpha_value = 0.4 if agent != "l2_er" else 1.0
            for seed_idx in range(seed_matrix.shape[0]):
                seed_epsilon_ranks = seed_matrix[seed_idx]
                
                valid_mask = ~np.isnan(task_accuracy)
                if np.any(valid_mask):
                    all_epsilon_ranks.extend(seed_epsilon_ranks[valid_mask])
                    all_accuracies.extend(task_accuracy[valid_mask])
                
                label = AGENT_LABELS.get(agent) if seed_idx == 0 else None
                plt.scatter(seed_epsilon_ranks, task_accuracy, 
                           s=20, alpha=alpha_value, label=label, color=color_hex)
        else:
            avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
            valid_mask = ~np.isnan(task_accuracy)
            if np.any(valid_mask):
                all_epsilon_ranks.extend(avg_epsilon_ranks[valid_mask])
                all_accuracies.extend(task_accuracy[valid_mask])
            
            alpha_value = 0.4 if agent != "l2_er" else 1.0
            plt.scatter(avg_epsilon_ranks, task_accuracy, 
                       s=20, alpha=alpha_value, label=AGENT_LABELS.get(agent), color=color_hex)

    # Calculate and plot linear regression line
    if len(all_epsilon_ranks) > 1:
        all_epsilon_ranks = np.array(all_epsilon_ranks)
        all_accuracies = np.array(all_accuracies)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_epsilon_ranks, all_accuracies)
        
        x_min, x_max = plt.xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        
        plt.plot(x_line, y_line, 'k--', alpha=0.8, linewidth=2, 
                label=f'Linear fit ($R^2$={r_value**2:.3f})')

    plt.xlabel(f"Percentage of eigenvalues outside $[-\\epsilon, +\\epsilon]$ ($\\epsilon$={epsilon_to_use:.2e})", fontsize=20)
    
    if dataset_type == "incremental_cifar" and reset_accuracy is not None:
        plt.ylabel("Accuracy Difference from RESET", fontsize=20)
    else:
        plt.ylabel("Accuracy", fontsize=20)
    
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    
    # Generate output filename
    epsilon_str = f"{epsilon_to_use:.2e}".replace("-", "neg")
    if dataset_type == "permuted_mnist":
        out_pdf = out_dir / f"permuted_mnist_epsilon_hessian_rank_vs_accuracy_{args.mode}_{args.phase}_eps{epsilon_str}.pdf"
    elif dataset_type == "incremental_cifar":
        out_pdf = out_dir / f"incremental_cifar_epsilon_hessian_rank_vs_accuracy_{args.mode}_{args.phase}_eps{epsilon_str}.pdf"
    else:
        out_pdf = out_dir / f"epsilon_hessian_rank_vs_accuracy_{args.mode}_{args.phase}_eps{epsilon_str}.pdf"
    
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[INFO] Saved plot: {out_pdf}")
    plt.close()

if __name__ == "__main__":
    main()
