import jax
import optax
import jax.numpy as jnp
from flax.training import train_state
from jax_convnet import ConvNet

# def create_train_state(rng, learning_rate, momentum, net_cls, dummy_input, num_classes=2):
#     net = net_cls(num_classes=num_classes)
#     params = net.init(rng, dummy_input)['params']
#     tx = optax.sgd(learning_rate, momentum)
#     return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

# @jax.jit
# def train_step(state, batch):
#     def loss_fn(params):
#         logits = state.apply_fn({'params': params}, batch['image'])
#         loss = optax.softmax_cross_entropy_with_integer_labels(
#             logits=logits, labels=batch['label']).mean()
#         return loss, logits
#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (loss, logits), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     return state, loss, logits

# @jax.jit
# def get_loss(params, x,y):
#     network = ConvNet(num_classes=2)
#     params = network.init(jax.random.PRNGKey(0), x)['params']
#     output, features = network.apply(params, x)
#     loss = optax.softmax_cross_entropy_with_integer_labels(
#             logits=output, labels=y).mean()
#     return loss

#  def loss_old(self, params, x, y):
#     output, features = self.network.apply(params, x)
#     loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
#     return loss

@jax.jit
def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)


class Agent:
    def __init__(self, network: ConvNet):
        self.network = network
        self.loss = jax.jit(self.loss)
    
    def predict(self, params, x):
        output = self.network.apply({'params': params}, x)
        return output
    
    def train_step(self, state, batch):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch['label']).mean()
            return loss, logits
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, logits
    

    def loss(self, params, x, y):
        output = self.network.apply({'params': params}, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
        return loss


class ShrinkAndPerturbAgent(Agent):
    def __init__(self, network: ConvNet, shrink_factor: float, perturb_scale: float):
        super().__init__(network)
        self.shrink_factor = shrink_factor
        self.perturb_scale = perturb_scale

    
    def perturb_params(self, params, rng, scale):
        """Add N(0, scale) noise to every parameter tensor in the tree."""
        
        leaves, treedef = jax.tree_util.tree_flatten(params)
        rngs = jax.random.split(rng, len(leaves))
        
        new_leaves = [
            p + scale * jax.random.normal(r, p.shape, p.dtype)
            for p, r in zip(leaves, rngs)
        ]
        return jax.tree_util.tree_unflatten(treedef, new_leaves), rngs[-1]


    def train_step(self, state, batch, rng):
        # Standard gradient update
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch['label']).mean()
            return loss, logits
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        # Shrink
        shrunk_params = jax.tree_util.tree_map(lambda p: p * (1.0 - self.shrink_factor), state.params)

        # Perturb
        rng, perturb_rng = jax.random.split(rng)
        perturbed_params, _ = perturb_params(shrunk_params, perturb_rng, self.perturb_scale)

        state = state.replace(params=perturbed_params)

        return state, loss, logits, rng




class EffectiveRankAgent:
    def __init__(self, network: ConvNet):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def predict(self, params, x):
        output, features = self.network.apply(params, x)
        return output, features
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, x):
        output, features = self.network.apply(params, x)
        erank_losses = [self.effective_rank(f) for f in features.values() if f is not None]
        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank

    def loss(self, params, x, y):
        output, features = self.network.apply(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
        return loss
    
    def perturb(self, params, perturb_scale, rng):
        return perturb_params(params, rng, perturb_scale)
