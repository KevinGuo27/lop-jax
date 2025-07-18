import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import nnx
from typing import Dict, Tuple, Optional

import time 



# Base Agent (SGD + L2)

@nnx.jit # we already have optax.chain() in the optimizer (this is L2)
def ce_plus_l2_loss(model, imgs, labels, current_classes):
    """Cross‑entropy and L2 reg. Returns (loss, logits)."""  
    logits_full, features = model(imgs)
    logits = logits_full[:, current_classes]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits

@nnx.jit
def base_update(model, opt, imgs, labels_int, current_classes):
    """One gradient step, returns (new_opt, loss, logits)."""
    (loss, logits), grads = nnx.value_and_grad(
        lambda model: ce_plus_l2_loss(model, imgs, labels_int, current_classes),
        has_aux=True
    )(model)

    opt.update(grads) # in-place update
    return opt, loss, logits

class BaseDLAgent:
    """SGD + L2; works directly on an *nnx.Optimizer* object."""
    def __init__(self, wd: float = 0.0):
        self.wd = wd

    def base_train_step(self, model, opt, imgs, labels_int, current_classes):
        # labels are already integers
        opt, loss, logits = base_update(model, opt, imgs, labels_int, current_classes)
        preds = jnp.argmax(logits, 1) 
        acc = jnp.mean(preds == labels_int)
        return opt, float(loss), float(acc)

# Effective Rank L2 Agent

def effective_rank(features, eps=1e-8):
    s = jnp.linalg.svdvals(features.T)
    s = jnp.abs(s)
    p = s / jnp.maximum(s.sum(), eps)
    entropy = -(p * jnp.log(p + eps)).sum()
    return jnp.exp(entropy)

@nnx.jit
def er_only_loss(model, imgs, labels_int, current_classes):
    logits_full, features = model(imgs, feature_list=[])
    logits = logits_full[:, current_classes]

    eranks = jnp.stack([effective_rank(f) for f in features])

    return -eranks.mean(), logits

@nnx.jit
def effective_rank_update(model, opt, imgs, labels_int, current_classes):
    (loss, logits), grads = nnx.value_and_grad(
        lambda model: er_only_loss(model, imgs, labels_int, current_classes),
        has_aux=True
    )(model)
    opt.update(grads)
    return opt, loss, logits

class EffectiveRankAgent(BaseDLAgent):

    def __init__(self,
                 er_batch: int = 25, # from train permuted mnist config file
                 er_steps: int = 1): # also from train permuted mnist config file
        super().__init__()
        self.er_batch = er_batch
        self.er_steps = er_steps

    def er_train_step(self, model, opt, imgs, labels_int, current_classes):
        opt, loss, logits = effective_rank_update(model, opt, imgs, labels_int, current_classes)
        preds = jnp.argmax(logits, axis=1)
        acc = jnp.mean(preds == labels_int)

        return opt, float(loss), float(acc)


class ShrinkPerturbAgent(BaseDLAgent):
    def __init__(self, wd: float = 0.0, shrink: float = 0.01, sigma: float = 1e-3):
        super().__init__()
        self.shrink = shrink
        self.sigma = sigma

    @nnx.jit
    def _update(self, opt, imgs, labels_int):
        opt, loss, logits = super()._update(opt, imgs, labels_int)

        # shrink
        params = nnx.state(opt.target, nnx.Param)
        shrunk = jtu.tree_map(lambda p: p * (1.0 - self.shrink), params)
        nnx.update(opt.model, {"params": shrunk})

        # perturb
        if self.sigma > 0.0:
            rng = jax.random.PRNGKey(opt.step)
            noisy = jtu.tree_map(lambda p: p + self.sigma * jax.random.normal(rng, p.shape), shrunk)
            nnx.update(opt.model, {"params": noisy})

        return opt, loss, logits



def build_agent(kind: str,
                *,
                shrink: float = 0.01,
                sigma: float = 1e-3,
                resgnt: Optional[object] = None):
    k = kind.lower()
    if k == 'base_dl':
        return BaseDLAgent()
    if k == 'cbp':
        return CBPAgent(resgnt)
    if k == 'shrink_perturb':
        return ShrinkPerturbAgent(shrink, sigma)
    if k == 'effective_rank':
        return EffectiveRankAgent()
    raise ValueError(f"Unknown agent kind: {kind}")