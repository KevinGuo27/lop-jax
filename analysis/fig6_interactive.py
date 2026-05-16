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
from matplotlib.widgets import Button
import subprocess
import os

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
    "l2_er": "L2 + ER",
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
        "default_data_root": Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/data"),
        "default_results_root": Path("/users/kguo32/data/kguo32/lop/imagenet/results"),
        "default_out_dir": Path("/users/kguo32/rl-opt/imagenet/hessian/plots"),
        "agent_results_map": {
            "bp": "bp_hessian",
            "cbp": "cbp_hessian", 
            "l2": "l2_hessian",
            "l2_er": "l2_er_hessian",
            "er": "er_hessian"
        },
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
        },
        "dataset_name": "Incremental CIFAR"
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
                
            print(f"Epsilon {eps:.2e}: R² = {r_squared:.4f}")
    
    print(f"Optimal epsilon: {optimal_epsilon:.2e} with R² = {best_r_squared:.4f}")
    return optimal_epsilon, best_r_squared

class InteractiveEpsilonRankPlot:
    def __init__(self, data_root, results_root, agents, mode, phase, epsilon_to_use, 
                 agent_results_map, dataset_type, out_dir):
        self.data_root = data_root
        self.results_root = results_root
        self.agents = agents
        self.mode = mode
        self.phase = phase
        self.epsilon_to_use = epsilon_to_use
        self.agent_results_map = agent_results_map
        self.dataset_type = dataset_type
        self.out_dir = out_dir
        
        # Store data for each point
        self.point_data = []
        self.selected_points = []
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Load RESET baseline for incremental CIFAR if needed
        self.reset_accuracy = None
        if dataset_type == "incremental_cifar":
            reset_results_dir = results_root / "reset_hessian"
            if not reset_results_dir.exists():
                reset_results_dir = results_root / "reset"
            if reset_results_dir.exists():
                self.reset_accuracy = load_accuracy_data(reset_results_dir)
                print(f"Loaded RESET baseline accuracy from: {reset_results_dir}")
        
        self.plot_data()
        self.setup_interactivity()
        
    def plot_data(self):
        """Plot the epsilon rank vs accuracy data"""
        all_epsilon_ranks = []
        all_accuracies = []
        
        for agent in self.agents:
            agent_dir = self.data_root / agent
            color_hex = AGENT_COLOR.get(agent, "black")
            
            print(f"Processing agent: {agent}")
            
            tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, self.mode, self.phase, self.epsilon_to_use)
            
            results_agent = self.agent_results_map.get(agent, agent)
            results_dir = self.results_root / results_agent
            accuracy = load_accuracy_data(results_dir)
            
            # Match task indices with accuracy
            if len(accuracy) >= len(tasks):
                task_accuracy = accuracy[tasks]
            else:
                task_accuracy = np.full(len(tasks), np.nan)
                task_accuracy[:len(accuracy)] = accuracy
            
            # For incremental CIFAR, calculate difference from RESET baseline
            if self.dataset_type == "incremental_cifar" and self.reset_accuracy is not None:
                if len(self.reset_accuracy) >= len(tasks):
                    reset_task_accuracy = self.reset_accuracy[tasks]
                else:
                    reset_task_accuracy = np.full(len(tasks), np.nan)
                    reset_task_accuracy[:len(self.reset_accuracy)] = self.reset_accuracy
                task_accuracy = task_accuracy - reset_task_accuracy
            
            # Plot data points
            if self.dataset_type == "incremental_cifar":
                alpha_value = 0.4 if agent != "l2_er" else 1.0
                for seed_idx in range(seed_matrix.shape[0]):
                    seed_epsilon_ranks = seed_matrix[seed_idx]
                    seed_epsilon_ranks = smooth_epsilon_ranks(seed_epsilon_ranks, sigma=1.0)
                    
                    valid_mask = ~np.isnan(task_accuracy)
                    if np.any(valid_mask):
                        all_epsilon_ranks.extend(seed_epsilon_ranks[valid_mask])
                        all_accuracies.extend(task_accuracy[valid_mask])
                    
                    # Store point data for interactivity
                    for i, (eps_rank, acc) in enumerate(zip(seed_epsilon_ranks, task_accuracy)):
                        if not np.isnan(acc):
                            self.point_data.append({
                                'x': eps_rank,
                                'y': acc,
                                'agent': agent,
                                'seed': seed_dirs[seed_idx].name,
                                'task': tasks[i],
                                'color': color_hex,
                                'alpha': alpha_value
                            })
                    
                    label = AGENT_LABELS.get(agent) if seed_idx == 0 else None
                    scatter = self.ax.scatter(seed_epsilon_ranks, task_accuracy, 
                                           s=20, alpha=alpha_value, label=label, color=color_hex)
            else:
                avg_epsilon_ranks = np.mean(seed_matrix, axis=0)
                valid_mask = ~np.isnan(task_accuracy)
                if np.any(valid_mask):
                    all_epsilon_ranks.extend(avg_epsilon_ranks[valid_mask])
                    all_accuracies.extend(task_accuracy[valid_mask])
                
                # Store point data for interactivity
                for i, (eps_rank, acc) in enumerate(zip(avg_epsilon_ranks, task_accuracy)):
                    if not np.isnan(acc):
                        self.point_data.append({
                            'x': eps_rank,
                            'y': acc,
                            'agent': agent,
                            'seed': 'average',
                            'task': tasks[i],
                            'color': color_hex,
                            'alpha': 0.4 if agent != "l2_er" else 1.0
                        })
                
                alpha_value = 0.4 if agent != "l2_er" else 1.0
                scatter = self.ax.scatter(avg_epsilon_ranks, task_accuracy, 
                                       s=20, alpha=alpha_value, label=AGENT_LABELS.get(agent), color=color_hex)
        
        # Calculate and plot linear regression line
        if len(all_epsilon_ranks) > 1:
            all_epsilon_ranks = np.array(all_epsilon_ranks)
            all_accuracies = np.array(all_accuracies)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_epsilon_ranks, all_accuracies)
            
            x_min, x_max = self.ax.get_xlim()
            x_line = np.linspace(x_min, x_max, 100)
            y_line = slope * x_line + intercept
            
            self.ax.plot(x_line, y_line, 'k--', alpha=0.8, linewidth=2, 
                        label=f'Linear fit ($R^2$={r_value**2:.3f})')
        
        self.ax.set_xlabel(f"Percentage of singular values greater than $\\epsilon$={self.epsilon_to_use:.2e}", fontsize=20)
        
        if self.dataset_type == "incremental_cifar" and self.reset_accuracy is not None:
            self.ax.set_ylabel("Accuracy Difference from RESET", fontsize=20)
        else:
            self.ax.set_ylabel("Accuracy", fontsize=20)
        
        self.ax.legend(fontsize=22)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Add instructions
        self.ax.text(0.02, 0.98, "Click on points to view hessian spectrum\nPress 's' to save selected points", 
                    transform=self.ax.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def setup_interactivity(self):
        """Setup mouse click and keyboard interactions"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def on_click(self, event):
        """Handle mouse clicks on the plot"""
        if event.inaxes != self.ax:
            return
            
        # Find the closest point
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
            
        min_dist = float('inf')
        closest_point = None
        
        for point in self.point_data:
            dist = np.sqrt((point['x'] - click_x)**2 + (point['y'] - click_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        
        if closest_point and min_dist < 0.01:  # Threshold for click detection
            self.select_point(closest_point)
            
    def select_point(self, point):
        """Select a point and show its hessian spectrum"""
        # Add to selected points if not already selected
        if point not in self.selected_points:
            self.selected_points.append(point)
            
            # Draw a circle around the selected point
            circle = patches.Circle((point['x'], point['y']), 0.005, 
                                  fill=False, edgecolor='red', linewidth=3)
            self.ax.add_patch(circle)
            
            print(f"\nSelected point:")
            print(f"  Agent: {point['agent']}")
            print(f"  Seed: {point['seed']}")
            print(f"  Task: {point['task']}")
            print(f"  Epsilon rank: {point['x']:.4f}")
            print(f"  Accuracy: {point['y']:.4f}")
            
            # Show hessian spectrum
            self.show_hessian_spectrum(point)
            
            self.fig.canvas.draw()
            
    def show_hessian_spectrum(self, point):
        """Display the hessian spectrum for the selected point"""
        agent = point['agent']
        seed = point['seed']
        task = point['task']
        
        # Construct path to hessian file
        if seed == 'average':
            # For average case, use the first seed directory
            seed_dirs = sorted([p for p in (self.data_root / agent).iterdir() 
                              if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
            if seed_dirs:
                seed_dir = seed_dirs[0]
            else:
                print(f"No seed directories found for agent {agent}")
                return
        else:
            seed_dir = self.data_root / agent / seed
            
        hessian_file = seed_dir / f"hessian_task_{task}_{self.phase}.npy"
        
        if not hessian_file.exists():
            print(f"Hessian file not found: {hessian_file}")
            return
            
        try:
            # Load and plot the spectrum
            grids, density = load_spectrum(hessian_file, self.mode)
            
            # Create a new figure for the spectrum
            fig_spectrum, ax_spectrum = plt.subplots(figsize=(10, 6))
            ax_spectrum.plot(grids, density, 'b-', linewidth=2)
            ax_spectrum.axvline(x=self.epsilon_to_use, color='r', linestyle='--', 
                               label=f'ε = {self.epsilon_to_use:.2e}')
            ax_spectrum.axvline(x=-self.epsilon_to_use, color='r', linestyle='--')
            ax_spectrum.set_xlabel('Eigenvalue', fontsize=14)
            ax_spectrum.set_ylabel('Density', fontsize=14)
            ax_spectrum.set_title(f'Hessian Spectrum - {AGENT_LABELS.get(agent, agent)} '
                                f'(Seed: {seed}, Task: {task})', fontsize=16)
            ax_spectrum.legend()
            ax_spectrum.grid(True, alpha=0.3)
            
            # Save the spectrum plot
            spectrum_filename = f"spectrum_{agent}_{seed}_task{task}_{self.mode}_{self.phase}.pdf"
            spectrum_path = self.out_dir / spectrum_filename
            fig_spectrum.savefig(spectrum_path, bbox_inches="tight")
            print(f"Saved spectrum plot: {spectrum_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error loading spectrum: {e}")
            
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 's':
            self.save_selected_points()
        elif event.key == 'c':
            self.clear_selection()
            
    def save_selected_points(self):
        """Save information about selected points"""
        if not self.selected_points:
            print("No points selected")
            return
            
        print(f"\nSaving {len(self.selected_points)} selected points...")
        
        # Save main plot with selected points highlighted
        epsilon_str = f"{self.epsilon_to_use:.2e}".replace("-", "neg")
        if self.dataset_type == "permuted_mnist":
            out_pdf = self.out_dir / f"interactive_permuted_mnist_epsilon_hessian_{self.mode}_{self.phase}_eps{epsilon_str}.pdf"
        elif self.dataset_type == "incremental_cifar":
            out_pdf = self.out_dir / f"interactive_incremental_cifar_epsilon_hessian_{self.mode}_{self.phase}_eps{epsilon_str}.pdf"
        else:
            out_pdf = self.out_dir / f"interactive_epsilon_hessian_{self.mode}_{self.phase}_eps{epsilon_str}.pdf"
        
        self.fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved interactive plot: {out_pdf}")
        
        # Print summary of selected points
        print("\nSelected Points Summary:")
        for i, point in enumerate(self.selected_points, 1):
            print(f"Point {i}: {AGENT_LABELS.get(point['agent'], point['agent'])} "
                  f"(Seed: {point['seed']}, Task: {point['task']}) - "
                  f"ε-rank: {point['x']:.4f}, Accuracy: {point['y']:.4f}")
                  
    def clear_selection(self):
        """Clear all selected points"""
        self.selected_points.clear()
        # Remove all circles
        for patch in self.ax.patches[:]:
            if isinstance(patch, patches.Circle):
                patch.remove()
        self.fig.canvas.draw()
        print("Selection cleared")

def main():
    parser = argparse.ArgumentParser(description="Interactive epsilon hessian rank vs accuracy plot")
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
    parser.add_argument("--epsilon", type=float, default=1e-2,
                        help="Fixed epsilon threshold")
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Sigma parameter for gaussian smoothing of epsilon ranks")

    args = parser.parse_args()
    
    # Set font to serif with Greek letter support
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'cmr10']

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

    print(f"Creating interactive plot for {dataset_name}")
    print(f"Using epsilon: {args.epsilon:.2e}")
    print("Instructions:")
    print("- Click on points to view their hessian spectrum")
    print("- Press 's' to save the plot with selected points")
    print("- Press 'c' to clear selection")

    # Create interactive plot
    plot = InteractiveEpsilonRankPlot(
        data_root, results_root, agents, args.mode, args.phase, 
        args.epsilon, agent_results_map, dataset_type, out_dir
    )
    
    plt.show()

if __name__ == "__main__":
    main()
