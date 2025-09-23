"""
Plot dead neuron analysis results across multiple experiments with error bands.
Aggregates results from all bundles processed by process_all_bundles.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
from collections import defaultdict

def find_all_results(results_dir: Path) -> Dict[str, List[Path]]:
    """Find all analysis results grouped by agent and experiment."""
    results = defaultdict(list)
    
    for agent_dir in results_dir.iterdir():
        if not agent_dir.is_dir():
            continue
            
        agent_name = agent_dir.name
        print(f"Found agent: {agent_name}")
        
def find_all_results(results_dir: Path) -> Dict[str, List[Path]]:
    """Find all analysis results grouped by agent and experiment."""
    results = defaultdict(list)
    
    for agent_dir in results_dir.iterdir():
        if not agent_dir.is_dir():
            continue
            
        agent_name = agent_dir.name
        print(f"Found agent: {agent_name}")
        
        # Look for dormant_analysis subdirectory
        dormant_analysis_dir = agent_dir / "dormant_analysis"
        if not dormant_analysis_dir.exists():
            print(f"No dormant_analysis directory found for {agent_name}")
            continue
            
        # Find all experiment result directories within dormant_analysis
        for exp_dir in dormant_analysis_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            # Check if this directory has our analysis results
            prev_file = exp_dir / "previous_tasks_dormant_units_analysis.npy"
            next_file = exp_dir / "next_task_dormant_units_analysis.npy"
            
            if prev_file.exists() and next_file.exists():
                results[agent_name].append(exp_dir)
                print(f"  Found results: {exp_dir.name}")
    print(f"All results found: {dict(results)}")
    return dict(results)

def load_experiment_data(exp_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load analysis data from a single experiment."""
    prev_file = exp_path / "previous_tasks_dormant_units_analysis.npy"
    next_file = exp_path / "next_task_dormant_units_analysis.npy"
    
    prev_dormant = np.load(prev_file)
    next_dormant = np.load(next_file)

    return prev_dormant, next_dormant

def aggregate_agent_data(agent_results: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate data across all experiments for one agent."""
    all_prev = []
    all_next = []
    num_tasks = None
    
    for exp_path in agent_results:
        print(f"Loading data from: {exp_path}")
        try:
            prev_dormant, next_dormant = load_experiment_data(exp_path)
            
            if num_tasks is None:
                num_tasks = len(prev_dormant)
            elif len(prev_dormant) != num_tasks:
                print(f"Warning: Inconsistent number of tasks in {exp_path}")
                continue
                
            all_prev.append(prev_dormant)
            all_next.append(next_dormant)
            
        except Exception as e:
            print(f"Error loading {exp_path}: {e}")
            continue
    
    if not all_prev:
        return None, None, None, None, 0
    
    # Stack all experiments
    all_prev = np.stack(all_prev, axis=0)  # (n_experiments, n_tasks)
    all_next = np.stack(all_next, axis=0)  # (n_experiments, n_tasks)
    
    # Compute mean and std
    prev_mean = np.mean(all_prev, axis=0)
    prev_std = np.std(all_prev, axis=0)
    next_mean = np.mean(all_next, axis=0)
    next_std = np.std(all_next, axis=0)
    
    return prev_mean, prev_std, next_mean, next_std, num_tasks

def plot_dormant_analysis(results_dict: Dict[str, List[Path]], output_dir: Path):
    """Create plots with error bands for all agents."""
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    agent_names = list(results_dict.keys())
    
    for i, (agent_name, agent_results) in enumerate(results_dict.items()):
        if not agent_results:
            continue
            
        print(f"\nProcessing agent: {agent_name} ({len(agent_results)} experiments)")
        
        prev_mean, prev_std, next_mean, next_std, num_tasks = aggregate_agent_data(agent_results)
        
        if prev_mean is None:
            print(f"No valid data for {agent_name}")
            continue
        
        color = colors[i % len(colors)]
        tasks = np.arange(num_tasks)
        
        # Plot previous tasks dormancy
        ax1.plot(tasks, prev_mean, color=color, label=f'{agent_name} (n={len(agent_results)})', linewidth=2)
        ax1.fill_between(tasks, prev_mean - prev_std, prev_mean + prev_std, 
                        color=color, alpha=0.2)
        
        # Plot next task dormancy  
        ax2.plot(tasks, next_mean, color=color, label=f'{agent_name} (n={len(agent_results)})', linewidth=2)
        ax2.fill_between(tasks, next_mean - next_std, next_mean + next_std,
                        color=color, alpha=0.2)
        
        print(f"Tasks: {num_tasks}")
        print(f"Prev dormancy final: {prev_mean[-1]:.4f} ± {prev_std[-1]:.4f}")
        print(f"Next dormancy final: {next_mean[-1]:.4f} ± {next_std[-1]:.4f}")
    
    # Formatting
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Dormant Unit Proportion')
    ax1.set_title('Previous Tasks Dormancy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim(0, 1)
    
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Dormant Unit Proportion')
    ax2.set_title('Next Task Dormancy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "dormant_units_comparison_with_error_bands.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.pdf'), bbox_inches='tight')
    
    print(f"\nSaved plots to: {plot_path}")
    
    # Also create individual plots for each metric
    create_individual_plots(results_dict, output_dir)

def create_individual_plots(results_dict: Dict[str, List[Path]], output_dir: Path):
    """Create separate plots for previous and next task dormancy."""
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Previous tasks plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (agent_name, agent_results) in enumerate(results_dict.items()):
        if not agent_results:
            continue
            
        prev_mean, prev_std, next_mean, next_std, num_tasks = aggregate_agent_data(agent_results)
        
        if prev_mean is None:
            continue
        
        color = colors[i % len(colors)]
        tasks = np.arange(num_tasks)
        
        ax.plot(tasks, prev_mean, color=color, label=f'{agent_name} (n={len(agent_results)})', linewidth=2)
        ax.fill_between(tasks, prev_mean - prev_std, prev_mean + prev_std, 
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Dormant Neuron Count')
    ax.set_title('Previous Tasks Dormancy Across Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(0, 1)
    
    plt.tight_layout()
    prev_path = output_dir / "previous_tasks_dormancy.png"
    plt.savefig(prev_path, dpi=300, bbox_inches='tight')
    plt.savefig(prev_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    # Next task plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (agent_name, agent_results) in enumerate(results_dict.items()):
        if not agent_results:
            continue
            
        prev_mean, prev_std, next_mean, next_std, num_tasks = aggregate_agent_data(agent_results)
        
        if next_mean is None:
            continue
        
        color = colors[i % len(colors)]
        tasks = np.arange(num_tasks)
        
        ax.plot(tasks, next_mean, color=color, label=f'{agent_name} (n={len(agent_results)})', linewidth=2)
        ax.fill_between(tasks, next_mean - next_std, next_mean + next_std,
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Dormant Neuron Count')
    ax.set_title('Next Task Dormancy Across Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(0, 1)
    
    plt.tight_layout()
    next_path = output_dir / "next_task_dormancy.png"
    plt.savefig(next_path, dpi=300, bbox_inches='tight')
    plt.savefig(next_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual plots to:")
    print(f"  {prev_path}")
    print(f"  {next_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot dormant neuron analysis with error bands")
    parser.add_argument("--results_dir", required=True, 
                       help="Directory containing agent subdirectories with analysis results")
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all results
    results_dict = find_all_results(results_dir)
    
    if not results_dict:
        print("No analysis results found!")
        return
    
    print(f"\nFound results for {len(results_dict)} agents:")
    for agent, exps in results_dict.items():
        print(f"  {agent}: {len(exps)} experiments")
    
    # Create plots
    plot_dormant_analysis(results_dict, output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

"""
usage:
python plot_dead_neurons_with_error_bands.py --results_dir ../results --output_dir ../results
"""