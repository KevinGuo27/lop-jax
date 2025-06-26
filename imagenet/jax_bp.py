import jax
import optax
import jax.numpy as jnp
from flax.training import train_state

def create_train_state(rng, learning_rate, momentum, net_cls, dummy_input):
    net = net_cls()
    params = net.init(rng, dummy_input)['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

@jax.jit
def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)
