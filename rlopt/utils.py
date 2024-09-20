import jax


def l2_regularization(params: dict, alpha: float = 0.001):
    return sum(
        alpha * (w ** 2).mean()
        for w in jax.tree_leaves(params["params"])
    )
