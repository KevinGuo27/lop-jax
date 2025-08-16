import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def gaussian_smooth1d(y, sigma_bins):
    if sigma_bins <= 0:
        return y
    radius = max(1, int(3 * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x**2) / (2.0 * sigma_bins**2))
    k /= k.sum()
    return np.convolve(y, k, mode="same")

def clean_density(density, clip="none", clip_hi_q=0.999, clip_k=1.5, z_thresh=8.0,
                  floor=0.0, smooth_sigma=0.0, eps=1e-12):
    d = np.asarray(density, dtype=float).copy()
    d[~np.isfinite(d)] = 0.0
    d = np.maximum(d, 0.0)

    if clip == "quantile":
        hi = np.quantile(d, clip_hi_q)
        d = np.minimum(d, hi)
    elif clip == "iqr":
        q1, q3 = np.quantile(d, [0.25, 0.75])
        iqr = q3 - q1
        cap = q3 + clip_k * iqr
        d = np.minimum(d, cap)
    elif clip == "zscore":
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + eps
        z = 0.6745 * (d - med) / mad
        d = np.where(z > z_thresh, med, d)

    if smooth_sigma > 0.0:
        d = gaussian_smooth1d(d, smooth_sigma)

    d = np.maximum(d, floor)
    return d

def _bin_masses(grids, density, eps=1e-12):
    grids = np.asarray(grids)
    density = np.asarray(density)
    dlam = np.diff(grids)
    dlam = np.clip(dlam, eps, None)
    dlam = np.concatenate([dlam, dlam[-1:]])
    p = np.maximum(density, 0.0) * dlam
    Z = p.sum()
    if not np.isfinite(Z) or Z <= eps:
        return None, None
    return p / Z, dlam

def trim_by_mass_quantiles(grids, density, q_lo=1e-3, q_hi=1-1e-3):
    """Keep bins whose cumulative mass lies in [q_lo, q_hi]."""
    p, dlam = _bin_masses(grids, density)
    if p is None:
        return grids, density
    cdf = np.cumsum(p)
    mask = (cdf >= q_lo) & (cdf <= q_hi)
    if mask.sum() < 3:   # avoid degenerate trimming
        return grids, density
    return grids[mask], density[mask]

def trim_by_value_zscore(grids, density, z_thresh=6.0):
    """Compute z-scores of eigenvalues under p-weighted mean/var; drop |z|>z_thresh."""
    p, dlam = _bin_masses(grids, density)
    if p is None:
        return grids, density
    lam = np.asarray(grids, dtype=float)
    mu = (p * lam).sum()
    var = (p * (lam - mu)**2).sum()
    sd = np.sqrt(max(var, 1e-12))
    z = (lam - mu) / sd
    mask = np.abs(z) <= z_thresh
    if mask.sum() < 3:
        return grids, density
    return grids[mask], density[mask]

def entropy_from_density(grids, density, *, base='e', eps=1e-12):
    grids = np.asarray(grids)
    density = np.asarray(density)
    if grids.ndim != 1 or density.ndim != 1 or grids.size != density.size or grids.size < 2:
        return np.nan

    # recompute bin widths AFTER trimming
    dlam = np.diff(grids)
    dlam = np.clip(dlam, eps, None)
    dlam = np.concatenate([dlam, dlam[-1:]])

    p = np.maximum(density, 0.0) * dlam
    Z = p.sum()
    if not np.isfinite(Z) or Z <= eps:
        return np.nan
    p = p / Z

    H = -(p * np.log(np.clip(p, eps, 1.0))).sum()
    if base == '2':
        H /= np.log(2.0)
    return H

if __name__ == "__main__":
    import argparse

    agents = ['bp', 'er', 'cbp', 'l2', 'l2_er']
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/users/kguo32/rl-opt/imagenet/hessian/data")
    parser.add_argument("--out_root", type=str, default="/users/kguo32/rl-opt/imagenet/hessian/entropy")
    parser.add_argument("--base", type=str, choices=['e', '2'], default='e')
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='test')

    # eigenvalue trimming (this is what filters out the outlier eigenvalues+density)
    parser.add_argument("--trim", type=str, choices=['none','quantile','zscore'], default='quantile')
    parser.add_argument("--q_lo", type=float, default=1e-3)   # keep 0.1%–99.9% by default
    parser.add_argument("--q_hi", type=float, default=1-1e-3)
    parser.add_argument("--z_thresh", type=float, default=6.0)

    # optional extra density cleanup (off by default)
    parser.add_argument("--clip", type=str, choices=['none','quantile','iqr','zscore'], default='none')
    parser.add_argument("--clip_hi_q", type=float, default=0.999)
    parser.add_argument("--clip_k", type=float, default=1.5)
    parser.add_argument("--dens_z_thresh", type=float, default=8.0)
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--floor", type=float, default=0.0)

    args = parser.parse_args()

    file_re = re.compile(r"hessian_task_(\d+)\.npy$")

    plt.figure(figsize=(8, 5))

    def load_trim_clean_and_entropy(payload, which, args):
        grids = payload.get(f'grids_{which}')
        dens  = payload.get(f'density_{which}')
        if grids is None or dens is None:
            return None
        # 1) trim outlier eigenvalues (and their density)
        if args.trim == 'quantile':
            grids, dens = trim_by_mass_quantiles(grids, dens, args.q_lo, args.q_hi)
        elif args.trim == 'zscore':
            grids, dens = trim_by_value_zscore(grids, dens, args.z_thresh)
        # 2) optional density cleanup
        dens = clean_density(dens, clip=args.clip, clip_hi_q=args.clip_hi_q,
                             clip_k=args.clip_k, z_thresh=args.dens_z_thresh,
                             floor=args.floor, smooth_sigma=args.smooth_sigma)
        # 3) entropy on trimmed spectrum
        return entropy_from_density(grids, dens, base=args.base)

    for agent_name in agents:
        data_dir = Path(args.data_root) / agent_name
        rows_train, rows_test = [], []

        for p in data_dir.glob("hessian_task_*.npy"):
            m = file_re.search(p.name)
            if not m:
                continue
            task = int(m.group(1))

            try:
                payload = np.load(p, allow_pickle=True).item()
            except Exception:
                continue

            H_tr = load_trim_clean_and_entropy(payload, 'train', args)
            H_te = load_trim_clean_and_entropy(payload, 'test', args)

            if H_tr is not None and np.isfinite(H_tr):
                rows_train.append((task, H_tr))
            if H_te is not None and np.isfinite(H_te):
                rows_test.append((task, H_te))

        if not rows_train and not rows_test:
            print(f"[WARN] No valid files for agent {agent_name} in {data_dir}.")
            continue

        if args.mode in ('train', 'both') and rows_train:
            rows_train.sort(key=lambda r: r[0])
            plt.plot([t for t,_ in rows_train], [h for _,h in rows_train],
                     label=f"{agent_name} (train)")

        if args.mode in ('test', 'both') and rows_test:
            rows_test.sort(key=lambda r: r[0])
            plt.plot([t for t,_ in rows_test], [h for _,h in rows_test],
                     label=f"{agent_name} (test)")

    ylabel = "Entropy (nats)" if args.base == 'e' else "Entropy (bits)"
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title("Permuted MNIST: entropy of Hessian spectral density (trimmed)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    suffix = "bits" if args.base == '2' else "nats"
    outname = f"hessian_entropy_all_agents_{args.mode}_trim-{args.trim}_{suffix}.pdf"
    combined_path = Path(args.out_root) / outname
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(combined_path)
    print(f"Saved plot to {combined_path}")
    plt.close()
