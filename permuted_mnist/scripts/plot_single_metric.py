#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===== Match the first script's color palette =====
colors = {
    'pink': '#ff96b6',
    'red': '#df5b5d',
    'orange': '#DD8453',
    'yellow': '#f8de7c',
    'green': '#3FC57F',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'white': '#ffffff',
    'light gray': '#d5d5d5',
    'dark gray': '#666666',
    'black': '#000000'
}

# Consistent agent→color mapping with your first script's legend order
AGENT_COLOR = {
    "l2_er": "green",
    "er": "cyan",
    "bp": "blue",
    "l2": "yellow",
    "cbp": "red",
}

FALLBACK_ORDER = ["green", "cyan", "blue", "yellow", "red", "purple", "orange", "brown", "black"]

FNAME_RE = re.compile(r"hessian_task_(\d+)_(init|end)\.npy$")

DEFAULT_AGENTS = ["bp", "er", "cbp", "l2_er", "l2"]


def weighted_stats(x, f):
    x = np.asarray(x)
    f = np.asarray(f)
    f = np.clip(f, a_min=0.0, a_max=None)
    Z = np.trapz(f, x)
    if Z <= 0 or not np.isfinite(Z):
        return np.nan, np.nan
    mean = np.trapz(x * f, x) / Z
    var = np.trapz((x - mean) ** 2 * f, x) / Z
    return mean, var


def load_spectrum(path, mode):
    d = np.load(path, allow_pickle=True).item()
    if mode == "train":
        return d["grids_train"], d["density_train"]
    elif mode == "test":
        return d["grids_test"], d["density_test"]
    else:
        raise ValueError("mode must be 'train' or 'test'")


def collect_phase_series(dir_with_files: Path, mode: str, phase: str):
    """Return (tasks_sorted, vars_sorted) for one directory that directly contains the .npy files for a given seed."""
    files = list(dir_with_files.glob("hessian_task_*_*.npy"))
    by_task = {}
    for p in files:
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        task = int(m.group(1))
        file_phase = m.group(2)
        if file_phase != phase:
            continue
        grids, dens = load_spectrum(p, mode)
        _, var = weighted_stats(grids, dens)
        by_task[task] = var
    if not by_task:
        return np.array([]), np.array([])
    tasks = np.array(sorted(by_task.keys()))
    vars_ = np.array([by_task[t] for t in tasks])
    return tasks, vars_


def nansem(a, axis=0):
    """SEM that ignores NaNs along the given axis."""
    a = np.asarray(a, dtype=float)
    valid = np.isfinite(a)
    count = valid.sum(axis=axis, keepdims=False)
    mean = np.nanmean(a, axis=axis)
    # broadcast mean back to a's shape along axis
    expanded_mean = np.expand_dims(mean, axis=axis)
    sq = np.where(valid, (a - expanded_mean) ** 2, 0.0)
    # sample variance with Bessel's correction where possible
    denom = np.maximum(count - 1, 1)
    var = sq.sum(axis=axis) / denom
    std = np.sqrt(var)
    sem = std / np.sqrt(np.maximum(count, 1))
    return sem


def gather_agent_seed_matrix(agent_dir: Path, mode: str, phase: str):
    """
    Returns:
      tasks: (T,) sorted unique tasks
      data:  (S, T) matrix of per-seed variance curves aligned on tasks (NaN where missing)
    """
    # Seeds are subdirs named as integers
    seed_dirs = [p for p in agent_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not seed_dirs:
        # Fall back: maybe files are directly under agent_dir (no per-seed subfolders)
        tasks, v = collect_phase_series(agent_dir, mode, phase)
        if tasks.size == 0:
            return np.array([]), np.empty((0, 0))
        return tasks, v[None, :]  # single "seed"

    # First collect each seed's (tasks, values)
    per_seed = []
    all_tasks = set()
    for sd in sorted(seed_dirs, key=lambda p: int(p.name)):
        t, v = collect_phase_series(sd, mode, phase)
        if t.size == 0:
            continue
        per_seed.append((t, v))
        all_tasks.update(t.tolist())

    if not per_seed:
        return np.array([]), np.empty((0, 0))

    tasks_sorted = np.array(sorted(all_tasks))
    T = tasks_sorted.size
    S = len(per_seed)
    data = np.full((S, T), np.nan, dtype=float)

    # Fill row per seed, aligning by task index
    task_to_idx = {t: i for i, t in enumerate(tasks_sorted)}
    for s_idx, (t, v) in enumerate(per_seed):
        for tt, vv in zip(t, v):
            data[s_idx, task_to_idx[int(tt)]] = float(vv)

    return tasks_sorted, data


def main():
    parser = argparse.ArgumentParser(description="Plot task vs Hessian variance for multiple agents with multi-seed mean ± SEM.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/users/kguo32/rl-opt/permuted_mnist/hessian/data"),
        help="Root directory containing per-agent folders with saved .npy files (optionally per-seed subfolders).",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="*",
        default=DEFAULT_AGENTS,
        help="Agents to include (folder names under data-root).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Use the train or test spectrum.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["init", "end"],
        default="init",
        help="Plot at either init or end.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to save plots. Default: /users/.../hessian/plots",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("/users/kguo32/rl-opt/permuted_mnist/hessian/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Style to match the first script =====
    plt.figure(figsize=(8, 5))  # same size
    linestyles = {"init": "-", "end": "--"}
    legend_labels = []

    for a_idx, agent in enumerate(args.agents):
        agent_dir = args.data_root / agent
        if not agent_dir.exists():
            print(f"[WARN] Missing agent folder: {agent_dir}")
            continue

        # Color selection
        color_key = AGENT_COLOR.get(agent, FALLBACK_ORDER[a_idx % len(FALLBACK_ORDER)])
        color_hex = colors[color_key]

        # Aggregate across seeds
        tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, args.mode, args.phase)
        if tasks.size == 0 or seed_matrix.size == 0:
            print(f"[WARN] No data for agent={agent} phase={args.phase}")
            continue

        # Compute across-seed statistics per task
        means = np.nanmean(seed_matrix, axis=0)
        # "variance across seeds" (per task) – could be saved/printed if needed
        seed_var = np.nanvar(seed_matrix, axis=0, ddof=1)  # sample variance across seeds
        errs = nansem(seed_matrix, axis=0)  # SEM for ribbon like your other script

        # Plot mean ± SEM
        plt.plot(
            tasks,
            means,
            linewidth=2.0,
            linestyle=linestyles.get(args.phase, "-"),
            label=agent,
            color=color_hex,
        )
        plt.fill_between(tasks, means - errs, means + errs, alpha=0.3, color=color_hex)

        legend_labels.append(agent)

    plt.xlabel("Task", fontsize=24)
    plt.ylabel("Variance", fontsize=24)

    if args.title:
        plt.title(args.title)

    if legend_labels:
        plt.legend()

    plt.tight_layout()

    out_pdf = out_dir / f"hessian_variance_{args.mode}_{args.phase}_multiseed.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[INFO] Saved plot: {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
