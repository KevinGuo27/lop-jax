import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


AGENT_COLORS = {
    "bp": "#3180df",
    "l2_er": "#3FC57F",
}


def load_spectrum(path: Path):
    data = np.load(path, allow_pickle=True).item()
    if "grids_train" in data and "density_train" in data:
        return np.array(data["grids_train"]), np.array(data["density_train"])
    if "grids" in data and "density" in data:
        return np.array(data["grids"]), np.array(data["density"])
    raise KeyError(f"Missing grids/density keys in {path}")


def plot_hessian_spectrum(ax, grids, density, label, color):
    ax.semilogy(grids, density, label=label, color=color, linewidth=2)


def main():
    parser = argparse.ArgumentParser(description="Plot Hessian spectra for Fig.1.")
    parser.add_argument("--tasks", type=int, nargs="*", default=[50, 300])
    parser.add_argument("--agents", type=str, nargs="*", default=["bp", "l2_er"])
    parser.add_argument("--phase", type=str, default="init", choices=["init", "end"])
    parser.add_argument("--out-name", type=str, default="fig1_hessian_spectra.pdf")
    args = parser.parse_args()

    fig_dir = Path(__file__).resolve().parent
    out_stem = Path(args.out_name).stem
    out_suffix = Path(args.out_name).suffix or ".pdf"
    agents_suffix = "_".join(args.agents)
    out_path = fig_dir / f"{out_stem}_{agents_suffix}{out_suffix}"

    nrows = len(args.agents)
    ncols = len(args.tasks)
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharey=True,
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, agent in enumerate(args.agents):
        for col_idx, task in enumerate(args.tasks):
            ax = axes[row_idx, col_idx]
            data_path = fig_dir / agent / f"hessian_task_{task}_{args.phase}.npy"
            grids, density = load_spectrum(data_path)
            plot_hessian_spectrum(
                ax,
                grids,
                density,
                label=agent.upper(),
                color=AGENT_COLORS.get(agent, "black"),
            )

            if row_idx == 0:
                ax.set_title(f"Task {task}")
            if col_idx == 0:
                ax.set_ylabel(f"{agent.upper()}\nDensity")

            ax.set_xlim(-100, 1000)
            ax.set_ylim(1e-10, 1e2)
            ax.set_xlabel("Eigenvalue")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[INFO] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
