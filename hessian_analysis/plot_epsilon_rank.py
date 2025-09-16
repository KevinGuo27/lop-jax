import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

colors = {
    'red': '#df5b5d',
    'orange': '#DD8453',
    'yellow': '#f8de7c',
    'green': '#3FC57F',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'black': '#000000'
}

AGENT_COLOR = {
    "l2_er": "green",
    "er": "cyan",
    "bp": "blue",
    "l2": "yellow",
    "cbp": "red",
}

FALLBACK_ORDER = ["green", "cyan", "blue", "yellow", "red", "purple", "orange", "brown", "black"]

FNAME_RE = re.compile(r"hessian_task_(\d+)_(init|end)\.npy$")

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
        },  # No mapping needed for ImageNet
        "dataset_name": "ImageNet"
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
        "dataset_name": "Permuted MNIST"
    },
    "incremental_cifar": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er"],
        "default_data_root": Path("/users/kguo32/rl-opt/incremental_cifar/hessian/data"),
        "default_results_root": Path("/users/kguo32/rl-opt/incremental_cifar/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/incremental_cifar/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian",
            "cbp": "cbp_hessian", 
            "l2": "l2_hessian",
            "l2_er": "l2_er_hessian",
            "er": "er_hessian"
        },  # No mapping needed for incremental CIFAR
        "dataset_name": "Incremental CIFAR"
    }
}

def load_spectrum(path, mode):
    """Load hessian spectrum data from .npy file"""
    d = np.load(path, allow_pickle=True).item()
    return (d["grids_train"], d["density_train"]) if mode == "train" else (d["grids_test"], d["density_test"])

def calculate_epsilon_hessian_rank(grids, density, eps=1e-1):
    """Calculate epsilon hessian rank: number of eigenvalues outside [-eps, +eps]"""
    # Count eigenvalues outside the epsilon range
    mask = np.abs(grids) > eps
    
    # Integrate the density over the range outside epsilon to get the count
    epsilon_count = np.trapz(density[mask], grids[mask])
    
    return epsilon_count

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
    data = np.stack(curves, axis=0)
    return tasks_ref, data

def load_accuracy_data(results_dir: Path):
    """Load accuracy data from best_hyperparam_per_env_res.pkl"""
    pkl_path = results_dir / "best_hyperparam_per_env_res.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Accuracy data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Extract accuracy scores - shape is typically (seeds, hyperparams, tasks, ...)
    scores = data['scores']
    
    # Take mean across seeds and hyperparams to get task-level accuracy
    if scores.ndim >= 3:
        # Average over seeds and hyperparams dimensions
        task_accuracy = np.mean(scores, axis=(0, 1))
        # Flatten any remaining dimensions
        task_accuracy = task_accuracy.flatten()
    else:
        task_accuracy = scores.flatten()
    
    return task_accuracy

def detect_dataset_type(data_root: Path) -> str:
    """Detect dataset type based on data root path"""
    data_root_str = str(data_root)
    if "imagenet" in data_root_str.lower():
        return "imagenet"
    elif "permuted_mnist" in data_root_str.lower() or "mnist" in data_root_str.lower():
        return "permuted_mnist"
    elif "incremental_cifar" in data_root_str.lower() or "cifar" in data_root_str.lower():
        return "incremental_cifar"
    else:
        # Default to imagenet if unclear
        return "imagenet"

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
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-1,
                        help="Epsilon threshold for counting eigenvalues within [-eps, +eps]")

    args = parser.parse_args()

    # Determine dataset type
    if args.dataset is None:
        if args.data_root is not None:
            dataset_type = detect_dataset_type(args.data_root)
        else:
            # Use default paths to detect
            dataset_type = "imagenet"  # Default fallback
    else:
        dataset_type = args.dataset

    # Get dataset configuration
    config = DATASET_CONFIGS[dataset_type]
    
    # Set defaults based on dataset
    data_root = args.data_root or config["default_data_root"]
    results_root = args.results_root or config["default_results_root"]
    agents = args.agents or config["default_agents"]
    out_dir = args.out_dir or config["default_out_dir"]
    agent_results_map = config["agent_results_map"]
    dataset_name = config["dataset_name"]

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for a_idx, agent in enumerate(agents):
        # Load hessian data
        agent_dir = data_root / agent
        color_key = AGENT_COLOR.get(agent, FALLBACK_ORDER[a_idx % len(FALLBACK_ORDER)])
        color_hex = colors[color_key]
        
        print(f"Processing agent: {agent}")
        print(f"Hessian data dir: {agent_dir}")
        
        tasks, seed_matrix = gather_agent_seed_matrix(
            agent_dir, args.mode, args.phase, args.epsilon
        )
        S = seed_matrix.shape[0]
        
        # Load accuracy data - use agent mapping if available
        if agent in agent_results_map:
            results_agent = agent_results_map[agent]
        else:
            results_agent = agent
        results_dir = results_root / results_agent
        print(f"Results dir: {results_dir}")
        accuracy = load_accuracy_data(results_dir)
        
        # Match task indices with accuracy
        # Assuming tasks are 0-indexed and accuracy array corresponds to tasks
        if len(accuracy) >= len(tasks):
            task_accuracy = accuracy[tasks]
        else:
            # If accuracy array is shorter, pad with NaN
            task_accuracy = np.full(len(tasks), np.nan)
            task_accuracy[:len(accuracy)] = accuracy
        
        # Calculate average epsilon ranks across all seeds for each task
        avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
        
        # Plot average points
        plt.scatter(avg_epsilon_ranks, task_accuracy, 
                   s=50, alpha=0.8, label=agent, color=color_hex)

    plt.xlabel(f"Epsilon Hessian Rank (ε={args.epsilon})", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    if args.title:
        plt.title(args.title, fontsize=18)
    else:
        plt.title(f"{dataset_name}: Epsilon Hessian Rank vs Accuracy ({args.mode}, {args.phase}, ε={args.epsilon})", fontsize=18)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Generate output filename based on dataset
    if dataset_type == "permuted_mnist":
        out_pdf = out_dir / f"permuted_mnist_epsilon_hessian_rank_vs_accuracy_{args.mode}_{args.phase}.pdf"
    elif dataset_type == "incremental_cifar":
        out_pdf = out_dir / f"incremental_cifar_epsilon_hessian_rank_vs_accuracy_{args.mode}_{args.phase}.pdf"
    else:
        out_pdf = out_dir / f"epsilon_hessian_rank_vs_accuracy_{args.mode}_{args.phase}.pdf"
    
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[INFO] Saved plot: {out_pdf}")
    plt.close()

if __name__ == "__main__":
    main()
