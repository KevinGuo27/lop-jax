import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as patches

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
    "l2_er": "L2-ER (Ours)",
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
        "dataset_name": "ImageNet"
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
    return tasks_ref, np.stack(curves, axis=0), seed_dirs

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

def plot_hessian_spectrum(grids, density, epsilon, agent, seed, task, mode, phase, out_dir):
    """Plot and save hessian spectrum"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grids, density, 'b-', linewidth=2)
    ax.axvline(x=epsilon, color='r', linestyle='--', 
               label=f'ε = {epsilon:.2e}')
    ax.axvline(x=-epsilon, color='r', linestyle='--')
    ax.set_xlabel('Eigenvalue', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Hessian Spectrum - {AGENT_LABELS.get(agent, agent)} '
                f'(Seed: {seed}, Task: {task})', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the spectrum plot
    spectrum_filename = f"spectrum_{agent}_{seed}_task{task}_{mode}_{phase}.pdf"
    spectrum_path = out_dir / spectrum_filename
    fig.savefig(spectrum_path, bbox_inches="tight")
    print(f"Saved spectrum plot: {spectrum_path}")
    
    plt.close(fig)
    return spectrum_path

def main():
    parser = argparse.ArgumentParser(description="Demo: Epsilon rank graph with two circled points and their hessian spectra")
    parser.add_argument("--data-root", type=Path, 
                        default=Path("/users/kguo32/rl-opt/imagenet/hessian/data"),
                        help="Root directory containing hessian data")
    parser.add_argument("--results-root", type=Path, 
                        default=Path("/users/kguo32/rl-opt/imagenet/results"),
                        help="Root directory containing results data")
    parser.add_argument("--out-dir", type=Path, 
                        default=Path("/users/kguo32/rl-opt/analysis"),
                        help="Output directory for plots")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--phase", type=str, choices=["init", "end"], default="init")
    parser.add_argument("--epsilon", type=float, default=1e-1,
                        help="Fixed epsilon threshold")

    args = parser.parse_args()
    
    # Set font to serif with Greek letter support
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'cmr10']

    config = DATASET_CONFIGS["imagenet"]
    agents = config["default_agents"]
    agent_results_map = config["agent_results_map"]
    dataset_name = config["dataset_name"]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating demo plot for {dataset_name}")
    print(f"Using epsilon: {args.epsilon:.2e}")

    # Create main figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect all data points for linear regression
    all_epsilon_ranks = []
    all_accuracies = []
    all_point_data = []  # Store all point information

    for agent in agents:
        agent_dir = args.data_root / agent
        color_hex = AGENT_COLOR.get(agent, "black")
        
        print(f"Processing agent: {agent}")
        
        tasks, seed_matrix, seed_dirs = gather_agent_seed_matrix(agent_dir, args.mode, args.phase, args.epsilon)
        
        results_agent = agent_results_map.get(agent, agent)
        results_dir = args.results_root / results_agent
        accuracy = load_accuracy_data(results_dir)
        
        # Match task indices with accuracy
        if len(accuracy) >= len(tasks):
            task_accuracy = accuracy[tasks]
        else:
            task_accuracy = np.full(len(tasks), np.nan)
            task_accuracy[:len(accuracy)] = accuracy
        
        # Calculate average epsilon ranks across seeds
        avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
        valid_mask = ~np.isnan(task_accuracy)
        if np.any(valid_mask):
            all_epsilon_ranks.extend(avg_epsilon_ranks[valid_mask])
            all_accuracies.extend(task_accuracy[valid_mask])
        
        # Store point data for potential selection
        for i, (eps_rank, acc) in enumerate(zip(avg_epsilon_ranks, task_accuracy)):
            if not np.isnan(acc):
                all_point_data.append({
                    'x': eps_rank,
                    'y': acc,
                    'agent': agent,
                    'task': tasks[i],
                    'color': color_hex
                })
        
        alpha_value = 0.4 if agent != "l2_er" else 1.0
        
        # Plot all regular points first
        ax.scatter(avg_epsilon_ranks, task_accuracy, 
                  s=20, alpha=alpha_value, label=AGENT_LABELS.get(agent), color=color_hex)

    # Calculate and plot linear regression line
    if len(all_epsilon_ranks) > 1:
        all_epsilon_ranks = np.array(all_epsilon_ranks)
        all_accuracies = np.array(all_accuracies)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_epsilon_ranks, all_accuracies)
        
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'k--', alpha=0.8, linewidth=2, 
                label=f'$R^2$={r_value**2:.3f}')

    ax.set_xlabel(f"Percentage of eigenvalues within $\\epsilon$={args.epsilon:.1f}", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.legend(fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Select BP points at task 100 (square) and task 1400 (triangle)
    target_task_square = 100
    target_task_triangle = 1400
    bp_points = [p for p in all_point_data if p['agent'] == 'bp']
    
    selected_points = []
    
    # Find BP point with task 100 (for square marker)
    bp_point_100 = None
    for point in bp_points:
        if point['task'] == target_task_square:
            bp_point_100 = point
            bp_point_100['marker'] = 's'  # square
            break
    
    # Find BP point with task 1400 (for triangle marker)
    bp_point_1400 = None
    for point in bp_points:
        if point['task'] == target_task_triangle:
            bp_point_1400 = point
            bp_point_1400['marker'] = '^'  # triangle
            break
    
    if bp_point_100:
        selected_points.append(bp_point_100)
        print(f"Selected BP point (square): Task {bp_point_100['task']}, ε-rank: {bp_point_100['x']:.4f}, Accuracy: {bp_point_100['y']:.4f}")
    else:
        print(f"No BP point found for task {target_task_square}")
    
    if bp_point_1400:
        selected_points.append(bp_point_1400)
        print(f"Selected BP point (triangle): Task {bp_point_1400['task']}, ε-rank: {bp_point_1400['x']:.4f}, Accuracy: {bp_point_1400['y']:.4f}")
    else:
        print(f"No BP point found for task {target_task_triangle}")
    
    # Plot the specific selected points with special markers on top
    for point in selected_points:
        if point['agent'] == 'bp':
            marker_style = point['marker']  # Use marker stored in point data
            marker_color = point['color']
        else:
            continue
            
        # Plot the special marker on top of the regular point
        ax.scatter(point['x'], point['y'], 
                  s=400, alpha=1.0, 
                  color=marker_color, marker=marker_style, 
                  edgecolors='black', linewidth=3, zorder=10)
    
    plt.tight_layout()
    
    # Save the main plot
    epsilon_str = f"{args.epsilon:.2e}".replace("-", "neg")
    out_pdf = out_dir / f"epsilon_hessian_rank.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved main plot: {out_pdf}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()
