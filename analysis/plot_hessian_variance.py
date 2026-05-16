import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
        "default_data_root": Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/data"),
        "default_out_dir": Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/plots"),
        "dataset_name": "ImageNet"
    },
    "permuted_mnist": {
        "default_agents": ["bp", "cbp", "l2", "l2_er", "er"],
        "default_data_root": Path("/users/kguo32/rl-opt/permuted_mnist/hessian/data"),
        "default_out_dir": Path("/users/kguo32/rl-opt/permuted_mnist/hessian/plots"),
        "dataset_name": "Permuted MNIST"
    },
    "incremental_cifar": {
        "default_agents": ["bp", "cbp", "l2"],
        "default_data_root": Path("/users/kguo32/rl-opt/incremental_cifar/hessian/data"),
        "default_out_dir": Path("/users/kguo32/rl-opt/incremental_cifar/hessian/plots"),
        "dataset_name": "Incremental CIFAR"
    }
}

DEFAULT_AGENTS = ["bp","cbp", "l2", "l2_er", "er"]

def weighted_stats(x, f):
    x = np.asarray(x)
    f = np.clip(np.asarray(f), a_min=0.0, a_max=None)
    Z = np.trapz(f, x)
    if not np.isfinite(Z) or Z <= 0:
        return np.nan, np.nan
    mean = np.trapz(x * f, x) / Z
    var  = np.trapz((x - mean) ** 2 * f, x) / Z
    return mean, var

def trim_by_grid_values(x, f, x_min=None, x_max=None, q_lower=None, q_upper=None):
    """
    Trim (x,f) by eigenvalue (grid) range.
    Priority: explicit x_min/x_max > percentile q_lower/q_upper.
      - x_min/x_max: absolute cutoffs in eigenvalue units.
      - q_lower/q_upper: percent cutoffs over the grid values (0..100).
    Returns trimmed (x_t, f_t). If trimming removes too much, fall back to original.
    """
    x = np.asarray(x)
    f = np.asarray(f)

    # Resolve thresholds
    lo, hi = x_min, x_max
    if lo is None or hi is None:
        # Use percentiles only if explicit thresholds not set
        if q_lower is not None:
            lo = np.percentile(x, q_lower)
        if q_upper is not None:
            hi = np.percentile(x, q_upper)

    # If still both None, nothing to do
    if lo is None and hi is None:
        return x, f

    if lo is None: lo = -np.inf
    if hi is None: hi =  np.inf
    if lo >= hi:
        # Degenerate; keep original
        return x, f

    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        # Too aggressive; keep original to avoid numerical issues
        return x, f

    x_t = x[mask]
    f_t = f[mask]

    # (Optional) re-normalize so the total mass is preserved relative to original
    Z  = np.trapz(np.clip(f,  a_min=0.0, a_max=None), x)
    Zt = np.trapz(np.clip(f_t, a_min=0.0, a_max=None), x_t)
    if np.isfinite(Z) and np.isfinite(Zt) and Zt > 0:
        f_t = f_t * (Z / Zt)
    return x_t, f_t

def load_spectrum(path, mode):
    d = np.load(path, allow_pickle=True).item()
    return (d["grids_train"], d["density_train"]) if mode == "train" else (d["grids_test"], d["density_test"])

def collect_phase_series(seed_dir: Path, mode: str, phase: str,
                         x_min, x_max, q_lower, q_upper):
    """Return (tasks_sorted, vars_sorted) for one seed directory, trimming by eigenvalue range."""
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
        grids, dens = load_spectrum(p, mode)
        gx, gf = trim_by_grid_values(grids, dens, x_min=x_min, x_max=x_max,
                                     q_lower=q_lower, q_upper=q_upper)
        _, var = weighted_stats(gx, gf)
        by_task[task] = var
    tasks = np.array(sorted(by_task.keys()))
    vars_ = np.array([by_task[t] for t in tasks])
    return tasks, vars_

def gather_agent_seed_matrix(agent_dir: Path, mode: str, phase: str,
                             x_min, x_max, q_lower, q_upper):
    seed_dirs = sorted([p for p in agent_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                       key=lambda p: int(p.name))
    if len(seed_dirs) == 0:
        raise FileNotFoundError(f"No seed subdirectories found in {agent_dir}")
    tasks_ref, v0 = collect_phase_series(seed_dirs[0], mode, phase, x_min, x_max, q_lower, q_upper)
    curves = [v0]
    for sd in seed_dirs[1:]:
        t, v = collect_phase_series(sd, mode, phase, x_min, x_max, q_lower, q_upper)
        if not np.array_equal(t, tasks_ref):
            raise ValueError(f"Task indices mismatch across seeds in {agent_dir}: {tasks_ref} vs {t}")
        curves.append(v)
    data = np.stack(curves, axis=0)
    return tasks_ref, data

def main():
    parser = argparse.ArgumentParser(description="Plot task vs Hessian variance with multi-seed mean ± SEM, with optional eigenvalue (grid) trimming.")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/data"))
    parser.add_argument("--agents", type=str, nargs="*", default=DEFAULT_AGENTS)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--phase", type=str, choices=["init", "end"], default="init")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)

    # === New: grid-based trimming options ===
    parser.add_argument("--x-min", type=float, default=None,
                        help="Minimum eigenvalue to keep (absolute cutoff).")
    parser.add_argument("--x-max", type=float, default=None,
                        help="Maximum eigenvalue to keep (absolute cutoff).")
    parser.add_argument("--x-q-lower", type=float, default=None,
                        help="Lower percentile (0..100) of grids to keep, used if --x-min is not provided.")
    parser.add_argument("--x-q-upper", type=float, default=None,
                        help="Upper percentile (0..100) of grids to keep, used if --x-max is not provided.")

    args = parser.parse_args()

    out_dir = args.out_dir or Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    linestyles = {"init": "-", "end": "--"}

    for a_idx, agent in enumerate(args.agents):
        agent_dir = args.data_root / agent
        color_key = AGENT_COLOR.get(agent, FALLBACK_ORDER[a_idx % len(FALLBACK_ORDER)])
        color_hex = colors[color_key]
        print(agent_dir)
        tasks, seed_matrix = gather_agent_seed_matrix(
            agent_dir, args.mode, args.phase,
            args.x_min, args.x_max, args.x_q_lower, args.x_q_upper
        )
        S = seed_matrix.shape[0]
        means = seed_matrix.mean(axis=0)
        seed_var = seed_matrix.var(axis=0, ddof=1) if S > 1 else np.zeros_like(means)
        errs = np.sqrt(seed_var) / np.sqrt(S) if S > 0 else np.zeros_like(means)

        plt.plot(tasks, means, linewidth=2.0,
                 linestyle=linestyles[args.phase], label=agent, color=color_hex)
        plt.fill_between(tasks, means - errs, means + errs, alpha=0.3, color=color_hex)

    plt.xlabel("Task", fontsize=24)
    plt.ylabel("Variance", fontsize=24)
    if args.title:
        plt.title(args.title)
    plt.legend()

    plt.tight_layout()
    out_pdf = out_dir / f"hessian_variance_{args.mode}_{args.phase}_multiseed.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[INFO] Saved plot: {out_pdf}")
    plt.close()

if __name__ == "__main__":
    main()
