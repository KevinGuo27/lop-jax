#!/usr/bin/env python3
"""Plot runtime comparison for Permuted MNIST experiments."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import orbax.checkpoint
from collections import defaultdict

# Matplotlib styling
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

AGENT_NAME_MAP = {
    'bp': 'BP', 'l2': 'L2', 'er': 'ER', 'l2_er': 'L2-ER', 
    'cbp': 'CBP', 'snp': 'SNP', 'snp_l2': 'SNP-L2', 
    'spectral_reg': 'Spectral Reg', 'laynorm_l2': 'LayerNorm-L2'
}

AGENT_COLORS = {
    'BP': '#1f77b4', 'L2': '#ff7f0e', 'ER': '#2ca02c', 'L2-ER': '#d62728',
    'CBP': '#9467bd', 'SNP': '#8c564b', 'SNP-L2': '#e377c2', 
    'Spectral Reg': '#7f7f7f', 'LayNorm+L2': '#bcbd22'
}


def load_results_from_study(study_path: Path):
    """Load all results from a study directory."""
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    runtimes, agent_name = [], None
    
    for result_dir in study_path.iterdir():
        if not result_dir.is_dir() or result_dir.name in ['ocdbt.process_0', 'd']:
            continue
        try:
            results = orbax_checkpointer.restore(result_dir)
            if 'total_runtime' in results:
                runtimes.append(float(results['total_runtime']))
            if 'args' in results and agent_name is None:
                agent_name = results['args'].get('agent', 'unknown')
        except Exception as e:
            print(f"Warning: Could not load {result_dir.name}: {e}")
    
    return agent_name, runtimes


def collect_all_runtimes(results_base_path: Path, study_names=None):
    """Collect runtimes for all algorithms."""
    all_runtimes = defaultdict(list)
    
    if study_names is None:
        study_names = [item.name for item in results_base_path.iterdir() 
                      if item.is_dir() and not item.name.startswith('.')]
    
    for study_name in study_names:
        study_path = results_base_path / study_name
        if not study_path.exists():
            continue
        
        agent_name, runtimes = load_results_from_study(study_path)
        if agent_name and runtimes:
            display_name = AGENT_NAME_MAP.get(agent_name, agent_name)
            all_runtimes[display_name].extend(runtimes)
    
    return all_runtimes


def plot_runtime_comparison(all_runtimes, output_path):
    """Create bar plot comparing runtimes."""
    algorithms, means, stderrs, colors = [], [], [], []
    
    sorted_items = sorted(all_runtimes.items(), key=lambda x: np.mean(x[1]))
    
    for agent_name, runtimes in sorted_items:
        if not runtimes:
            continue
        algorithms.append(agent_name)
        means.append(np.mean(runtimes) / 60.0)  # Convert to minutes
        stderrs.append(np.std(runtimes) / np.sqrt(len(runtimes)) / 60.0)
        colors.append(AGENT_COLORS.get(agent_name, '#333333'))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(algorithms))
    bars = ax.bar(x, means, yerr=stderrs, capsize=5, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Methods', fontsize=20, fontweight='bold')
    ax.set_ylabel('Total Runtime (minutes)', fontsize=20, fontweight='bold')
    ax.set_title('Runtime Comparison: Permuted MNIST', fontsize=24, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for bar, mean_val, stderr_val in zip(bars, means, stderrs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + stderr_val,
                f'{mean_val:.1f}m', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def print_summary(all_runtimes):
    """Print runtime summary table."""
    sorted_items = sorted(all_runtimes.items(), key=lambda x: np.mean(x[1]))
    
    print("\n" + "="*60)
    print(f"{'Algorithm':<15} {'Mean (min)':<12} {'Std (min)':<12} {'Runs':<8}")
    print("-"*60)
    for agent_name, runtimes in sorted_items:
        if not runtimes:
            continue
        print(f"{agent_name:<15} {np.mean(runtimes)/60:<12.1f} "
              f"{np.std(runtimes)/60:<12.1f} {len(runtimes):<8}")
    print("="*60)


def main():
    results_base_path = Path('/users/kguo32/rl-opt/rlopt/results')
    
    study_names = [
        'bp', 'l2', 'er',
        'l2_er', 'cbp_l2', 
        'spectral_reg', 'layernorm_l2'
    ]
    
    all_runtimes = collect_all_runtimes(results_base_path, study_names)
    
    if not all_runtimes:
        print("Error: No runtime data found!")
        return
    
    print_summary(all_runtimes)
    
    output_path = Path('/users/kguo32/rl-opt/analysis/rl_runtime_comparison.pdf')
    plot_runtime_comparison(all_runtimes, output_path)
    print("✓ Analysis complete!")


if __name__ == "__main__":
    main()
