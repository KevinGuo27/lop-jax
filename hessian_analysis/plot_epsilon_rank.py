#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FNAME_RE = re.compile(r"hessian_task_(\d+)_(init|end)\.npy$")
DEFAULT_AGENTS = ["bp", "er", "cbp", "l2_er", "l2"]

def load_spectrum(path, mode):
    d = np.load(path, allow_pickle=True).item()
    if mode == "train":
        return d["grids_train"], d["density_train"]
    elif mode == "test":
        return d["grids_test"], d["density_test"]
    else:
        raise ValueError("mode must be 'train' or 'test'")

def mass_outside_band(x, f, lo, hi, absolute=False):
    """
    Return mass OUTSIDE [lo, hi] from spectral density (x: eigenvalue grid, f: density).
    If absolute=False, return FRACTION of total mass; if True, return ABSOLUTE mass.
    Uses |f| as requested.
    """
    x = np.asarray(x)
    f = np.abs(np.asarray(f))  # use abs per user preference

    total = np.trapz(f, x)
    if not np.isfinite(total) or total <= 0:
        return np.nan

    # normalize band and clamp to grid range
    lo_, hi_ = (lo, hi) if lo <= hi else (hi, lo)
    lo_ = max(lo_, x.min())
    hi_ = min(hi_, x.max())

    if lo_ >= hi_:
        # if band collapses or is outside grid, "outside band" = all mass
        return total if absolute else 1.0

    in_mask = (x >= lo_) & (x <= hi_)
    mass_in = np.trapz(f[in_mask], x[in_mask]) if np.any(in_mask) else 0.0
    mass_out = total - mass_in
    return mass_out if absolute else (mass_out / total)

def collect_phase_series(agent_dir: Path, mode: str, phase: str,
                         eps: float, window: float, eps_range, absolute: bool):
    """Return (tasks_sorted, values_sorted) for one agent and phase."""
    files = list(agent_dir.glob("hessian_task_*_*.npy"))
    by_task = {}
    for p in files:
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        task = int(m.group(1))
        p_phase = m.group(2)
        if p_phase != phase:
            continue
        try:
            grids, dens = load_spectrum(p, mode)
            if eps_range is not None:
                lo, hi = eps_range
            else:
                lo, hi = eps - window, eps + window
            val = mass_outside_band(grids, dens, lo, hi, absolute=absolute)
        except Exception as e:
            print(f"[WARN] {agent_dir.name} {phase} failed on {p}: {e}")
            val = np.nan
        by_task[task] = val
    tasks = sorted(by_task.keys())
    vals = [by_task[t] for t in tasks]
    return np.array(tasks), np.array(vals)

def main():
    parser = argparse.ArgumentParser(
        description="Plot task vs number/fraction of eigenvalues OUTSIDE an ε-range for multiple agents."
    )
    parser.add_argument("--data-root", type=Path,
                        default=Path("/users/kguo32/rl-opt/permuted_mnist/hessian/data"),
                        help="Root directory containing per-agent folders with saved .npy files.")
    parser.add_argument("--agents", type=str, nargs="*", default=DEFAULT_AGENTS,
                        help="Agents to include (folder names under data-root).")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="Use the train or test spectrum.")
    parser.add_argument("--phase", type=str, choices=["init", "end", "both"], default="both",
                        help="Plot at init, end, or both.")

    # ε-band controls
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="Center ε for the eigenvalue band (ignored if --eps-range is set).")
    parser.add_argument("--window", type=float, default=1e-2,
                        help="Half-width for symmetric band [ε - window, ε + window]. Ignored if --eps-range is set.")
    parser.add_argument("--eps-range", type=float, nargs=2, default=None,
                        help="Explicit eigenvalue range [low high]. Overrides --epsilon/--window.")

    parser.add_argument("--absolute", action="store_true",
                        help="Plot absolute mass OUTSIDE the band (≈ #eigs if density is count-normalized).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save plots/CSVs. Default: /users/.../hessian/plots/_multi")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    parser.add_argument("--show", action="store_true", help="Display the window in addition to saving.")
    args = parser.parse_args()

    phases = ["init", "end"] if args.phase == "both" else [args.phase]

    out_dir = args.out_dir or Path("/users/kguo32/rl-opt/permuted_mnist/hessian/plots/_multi")
    out_dir.mkdir(parents=True, exist_ok=True)

    # style cycles
    markers = ["o", "s", "D", "^", "v", "P", "X", "<", ">"]
    linestyles = {"init": "-", "end": "--"}

    plt.figure(figsize=(10, 6))
    legend_labels = []

    # Range description for labels/filenames
    if args.eps_range is not None:
        lo, hi = args.eps_range
        band_desc = f"[{lo:g},{hi:g}]"
        band_file = f"{lo:g}_{hi:g}".replace('.', 'p')
    else:
        lo, hi = args.epsilon - args.window, args.epsilon + args.window
        band_desc = f"[{args.epsilon:g}±{args.window:g}]"
        band_file = f"eps{args.epsilon:g}_w{args.window:g}".replace('.', 'p')

    # Collect & plot
    for a_idx, agent in enumerate(args.agents):
        agent_dir = Path(args.data_root) / agent
        if not agent_dir.exists():
            print(f"[WARN] Missing agent folder: {agent_dir}")
            continue

        for phase in phases:
            tasks, vals = collect_phase_series(
                agent_dir, args.mode, phase,
                eps=args.epsilon, window=args.window,
                eps_range=args.eps_range, absolute=args.absolute
            )
            if tasks.size == 0:
                print(f"[WARN] No data for agent={agent} phase={phase}")
                continue

            # Save per-(agent, phase) CSV
            metric_kind = "abs" if args.absolute else "frac"
            csv_path = out_dir / f"{agent}_eigs_OUTSIDE_band_{band_file}_{args.mode}_{phase}_{metric_kind}.csv"
            np.savetxt(csv_path, np.column_stack([tasks, vals]),
                       delimiter=",", header="task,eigs_outside_band", comments="")
            print(f"[INFO] Saved: {csv_path}")

            label = f"{agent} ({phase})" if len(phases) > 1 else f"{agent}"
            plt.plot(
                tasks,
                vals,
                marker=markers[a_idx % len(markers)],
                linewidth=1.8,
                linestyle=linestyles.get(phase, "-"),
                label=label,
            )
            legend_labels.append(label)

    plt.xlabel("Task Number")
    yname = "Eigenvalue Count OUTSIDE Band" if args.absolute else "Fraction OUTSIDE Band"
    plt.ylabel(f"{yname} {band_desc}")
    default_title = f"Eigenvalues OUTSIDE {band_desc} ({args.mode}) — " + ", ".join(args.agents)
    plt.title(args.title or default_title)
    if legend_labels:
        plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_pdf = out_dir / f"eigs_OUTSIDE_band_{band_file}_{args.mode}_{args.phase}_{'abs' if args.absolute else 'frac'}.pdf"
    plt.savefig(out_pdf)
    print(f"[INFO] Saved plot: {out_pdf}")
    if args.show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()
