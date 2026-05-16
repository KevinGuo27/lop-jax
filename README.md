# Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning

Code for **"Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning"** (ICML 2026, PMLR 306).

**Authors:** Arjun Prakash\*, Naicheng He\*, Kaicheng Guo\*, Saket Tiwari, Tyrone Serapio, Ruo Yu Tao, Amy Greenwald, George Konidaris

We show that **Hessian spectral collapse** -- the degeneration of the loss landscape's curvature spectrum -- is the central mechanism behind plasticity loss in continual learning. We propose **L2-ER**, a simple regularizer combining L2 weight decay with an effective rank penalty, that directly prevents spectral collapse and preserves plasticity across diverse benchmarks.

## Performance

L2-ER maintains plasticity across all four environments. Classification accuracy is reported for supervised benchmarks; online returns for RL.

![Performance across all environments](analysis/performance_comparison_2x2.png)

## Hessian Epsilon-Rank vs. Accuracy

Across tasks on Continual ImageNet, training accuracy is positively correlated with the epsilon-rank of the Hessian (R^2 = 0.711). L2-ER (blue) preserves high epsilon-rank; BP (red) collapses.

![Epsilon-rank scatter plot](analysis/epsilon_hessian_rank.png)

## Benchmarks

| Benchmark | # Tasks | Model | Metric |
|---|---|---|---|
| Permuted MNIST | 800 | 3-layer MLP (width 1000) | Accuracy |
| Continual ImageNet | 2000 | ResNet-18 | Accuracy |
| Incremental CIFAR | 20 | ResNet-18 | Accuracy |
| Slippery Ant (RL) | 5 env changes | Actor-Critic MLP (width 256) | Online returns |

## Algorithms

| Method | `--agent` key | Description |
|---|---|---|
| Backpropagation | `bp` | Standard baseline |
| L2 Regularization | `l2` | L2 weight decay |
| Effective Rank | `er` | Maximizes effective rank of activations |
| **L2-ER (Ours)** | `l2_er` | L2 + effective rank penalty |
| Continual Backprop | `cbp` | Replaces low-utility neurons |
| LayerNorm + L2 | `laynorm_l2` | Layer normalization with L2 |
| Spectral Reg | `spectral_reg` | Regularizes weight singular values |
| Reset | `reset` | Reinitializes at task change (CIFAR only) |

## Installation

```bash
git clone https://github.com/<user>/lop-jax.git
cd lop-jax
pip install -e .
```

Dependencies are in `requirements.txt`. Requires JAX with CUDA 12, PyTorch (data loading), Flax, Optax, and Brax (RL).

## Usage

### Training

```bash
# Permuted MNIST
python -m permuted_mnist.train_permuted_mnist --agent l2_er --lr 0.01 --weight_decay 0.001 --seed 2025

# Continual ImageNet
python -m imagenet.train_imagenet --agent l2_er --lr 0.01 --weight_decay 0.001 --seed 2025

# Incremental CIFAR
python -m incremental_cifar.train_incremental_cifar --agent l2_er --lr 0.1 --weight_decay 0.0005 --seed 2025

# Slippery Ant (RL)
python -m rlopt.ppo_nonstationary --agent l2_er --lr 0.00025 --seed 2025
```

### Hessian Computation

Add `--compute_hessian --compute_hessian_interval 100` to any training command. Eigenspectra are saved to `hessian/data/` and plots to `hessian/plots/`.

### SLURM Sweep Pipeline

1. Define hyperparameters in `<benchmark>/scripts/hyperparams/<agent>.py`
2. `python write_jobs.py` to generate run files
3. `sbatch multiple_slurm_jobs.sh` to submit
4. `python parse_experiments.py ../results` to aggregate
5. `python plot_single_metric.py` to plot

## Repository Structure

```
lop-jax/
├── permuted_mnist/          # Permuted MNIST benchmark
├── imagenet/                # Continual ImageNet benchmark
├── incremental_cifar/       # Incremental CIFAR-100 benchmark
├── rlopt/                   # Slippery Ant RL benchmark
├── analysis/                # Hessian analysis & paper figures
├── results/                 # Experiment outputs
├── requirements.txt
└── setup.py
```

Each benchmark contains: `train_*.py` (training), `config.py` (hyperparameters), `cbp.py` (Continual Backprop), `scripts/` (job generation & plotting), and `utils/` (Hessian computation, Lanczos algorithm, evaluation metrics).

## Citation

```bibtex
@inproceedings{prakash2026spectral,
  title     = {Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning},
  author    = {Prakash, Arjun and He, Naicheng and Guo, Kaicheng and Tiwari, Saket and Serapio, Tyrone and Tao, Ruo Yu and Greenwald, Amy and Konidaris, George},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {306},
  year      = {2026}
}
```
