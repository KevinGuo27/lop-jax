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
DEFAULT_AGENTS = ["bp", "cbp", "l2"]


def weighted_stats(x, f):
    x = np.asarray(x)
    f = np.asarray(f)
    Z = np.trapz(f, x)
    mean = np.trapz(x * f, x) / Z
    var = np.trapz((x - mean) ** 2 * f, x) / Z
    return mean, var


def load_spectrum(path, mode):
    d = np.load(path, allow_pickle=True).item()
    return (d["grids_train"], d["density_train"]) if mode == "train" else (d["grids_test"], d["density_test"])


def collect_phase_series(seed_dir: Path, mode: str, phase: str):
    """Return (tasks_sorted, vars_sorted) for one seed directory."""
    files = list(seed_dir.glob("hessian_task_*_*.npy"))
    by_task = {}
    for p in files:
        m = FNAME_RE.search(p.name)
        task = int(m.group(1))
        file_phase = m.group(2)
        if file_phase != phase:
            continue
        grids, dens = load_spectrum(p, mode)
        _, var = weighted_stats(grids, dens)
        by_task[task] = var
    tasks = np.array(sorted(by_task.keys()))
    vars_ = np.array([by_task[t] for t in tasks])
    return tasks, vars_


def gather_agent_seed_matrix(agent_dir: Path, mode: str, phase: str):
    """Returns tasks (T,), data (S, T). Assumes identical tasks/order across seeds."""
    seed_dirs = sorted([p for p in agent_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                       key=lambda p: int(p.name))
    tasks_ref, v0 = collect_phase_series(seed_dirs[0], mode, phase)
    curves = [v0]
    for sd in seed_dirs[1:]:
        t, v = collect_phase_series(sd, mode, phase)
        curves.append(v)
    data = np.stack(curves, axis=0)  # (S, T)
    return tasks_ref, data


def main():
    parser = argparse.ArgumentParser(description="Plot task vs Hessian variance with multi-seed mean ± SEM.")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/users/kguo32/rl-opt/imagenet/hessian/data"))
    parser.add_argument("--agents", type=str, nargs="*", default=DEFAULT_AGENTS)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--phase", type=str, choices=["init", "end"], default="init")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path("/users/kguo32/rl-opt/imagenet/hessian/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    linestyles = {"init": "-", "end": "--"}

    for a_idx, agent in enumerate(args.agents):
        agent_dir = args.data_root / agent

        color_key = AGENT_COLOR.get(agent, FALLBACK_ORDER[a_idx % len(FALLBACK_ORDER)])
        color_hex = colors[color_key]
        print(agent_dir)
        tasks, seed_matrix = gather_agent_seed_matrix(agent_dir, args.mode, args.phase)
        S = seed_matrix.shape[0]
        means = seed_matrix.mean(axis=0)
        seed_var = seed_matrix.var(axis=0)
        errs = np.sqrt(seed_var) / np.sqrt(S)

        plt.plot(tasks, means, linewidth=2.0,
                 linestyle=linestyles[args.phase], label=agent, color=color_hex)
        plt.fill_between(tasks, means - errs, means + errs, alpha=0.3, color=color_hex)

    plt.xlabel("Task", fontsize=24)
    plt.ylabel("Variance", fontsize=24)
    plt.ylim(0, 250)
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
