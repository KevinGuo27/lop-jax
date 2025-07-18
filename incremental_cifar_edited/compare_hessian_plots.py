"""
Script to compare Hessian spectrum plots across algorithms.

This script scans the sub-directories inside the provided `hessian` folder (each
sub-directory corresponds to a learning algorithm, e.g. `l2`, `bp`, …).  It then
collects the PNG files generated while training, whose naming convention is::

    hessian_task_{task_number}_{stage}.png

where `stage` is typically `at_init` or `end`.

Given a list of tasks (e.g. ``0,5,10,20,100,150,200,300,400,500,n``) and a
chosen *stage* (default: ``at_init``) the script creates a grid where **every
row corresponds to an algorithm** and **every column to a task**.  The last
special task name ``n`` always refers to the *largest* task id found for the
respective algorithm.

Usage
-----
python compare_hessian_plots.py --hessian_dir permuted_mnist/hessian \\
                                --tasks 0,5,10,20,100,150,200,300,400,500,n \\
                                --stage at_init \\
                                --outfile comparison.png

This will save the resulting figure to ``comparison.png`` and show it
interactively if executed in an environment with a display available.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from PIL import Image


_TASK_RE = re.compile(r"hessian_task_(\d+)_([a-z_]+)\.png$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Hessian plots across algorithms.")
    parser.add_argument(
        "--hessian_dir",
        type=Path,
        default=Path(__file__).parent / "hessian",
        help="Path to the folder containing algorithm sub-directories.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="0,5,10,20, 50,100,150,200,300,400,500,600",
        help="Comma-separated list of task indices to include. Use 'n' for the last available task.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="at_init",
        help="Which stage to visualise (e.g., 'at_init' or 'end').",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("comparison.png"),
        help="Where to save the resulting figure.",
    )
    return parser.parse_args()


def _extract_tasks(algorithm_dir: Path, stage: str) -> dict[int, Path]:
    """Return mapping from *task id* to corresponding image path for a given stage."""
    mapping: dict[int, Path] = {}
    for png in algorithm_dir.glob(f"hessian_task_*_{stage}.png"):
        m = _TASK_RE.match(png.name)
        if m and m.group(2) == stage:
            task_id = int(m.group(1))
            mapping[task_id] = png
    return mapping


def _resolve_tasks(requested: Sequence[str], available: List[int]) -> List[int]:
    """Translate user-requested task tokens to concrete integer task ids.

    The special token ``'n'`` resolves to ``max(available)``.
    """
    resolved: List[int] = []
    if not available:
        return resolved

    last = max(available)

    for tok in requested:
        tok = tok.strip()
        if tok.lower() == "n":
            resolved.append(last)
        else:
            try:
                resolved.append(int(tok))
            except ValueError:
                print(f"[WARN] Ignoring invalid task token: {tok}")
    return resolved


def _load_image(path: Path | None) -> Image.Image:
    """Return a Pillow image; produce a blank placeholder if *path* is None."""
    if path and path.exists():
        return Image.open(path)
    # Create a blank white placeholder
    return Image.new("RGB", (640, 480), color="white")


def main() -> None:
    args = _parse_args()

    if not args.hessian_dir.exists():
        raise FileNotFoundError(f"Hessian directory not found: {args.hessian_dir}")

    # Identify algorithms
    alg_dirs = [p for p in args.hessian_dir.iterdir() if p.is_dir()]
    alg_dirs.sort(key=lambda p: p.name)

    if not alg_dirs:
        raise RuntimeError(f"No algorithm sub-directories found inside {args.hessian_dir}")

    # Prepare tasks list (strings) once
    requested_tasks = [tok.strip() for tok in args.tasks.split(",") if tok.strip()]

    # Collect images for each algorithm
    alg_to_tasks_paths: dict[str, dict[int, Path]] = {}
    all_resolved_tasks: set[int] = set()
    for alg_dir in alg_dirs:
        available_mapping = _extract_tasks(alg_dir, args.stage)
        resolved_tasks = _resolve_tasks(requested_tasks, list(available_mapping.keys()))
        # Keep only tasks that actually exist
        existing_tasks = {t: available_mapping.get(t) for t in resolved_tasks if t in available_mapping}
        alg_to_tasks_paths[alg_dir.name] = existing_tasks
        all_resolved_tasks.update(existing_tasks.keys())

    # Sort tasks globally so that columns align across algorithms
    sorted_tasks = sorted(all_resolved_tasks)
    if not sorted_tasks:
        raise RuntimeError("None of the requested tasks were found in any algorithm directory.")

    n_rows = len(alg_dirs)
    n_cols = len(sorted_tasks)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)

    for row_idx, alg_dir in enumerate(alg_dirs):
        alg_name = alg_dir.name
        task_to_path = alg_to_tasks_paths[alg_name]
        for col_idx, task in enumerate(sorted_tasks):
            ax = axes[row_idx][col_idx]
            img_path = task_to_path.get(task)
            img = _load_image(img_path)
            ax.imshow(img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"Task {task}")
            if col_idx == 0:
                ax.set_ylabel(alg_name)

    fig.tight_layout()
    fig.savefig(args.outfile, dpi=300)
    print(f"Saved comparison figure to {args.outfile.resolve()}")

    # Try to display interactively (won't error in headless environments)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main() 