import distrax
import jax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np
import functools

class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )

class SimpleNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out

class Actor(nn.Module):
    action_dim: int
    continuous: bool = False
    hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        h1 = nn.Dense(2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="a_0")(x)
        h1 = act_fn(h1)

        activations = {
            "a_0": None,
            "a_1": h1,
        }

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="a_1")(h1)

        if self.continuous:
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            activations["log_std"] = None
            pi = distrax.MultivariateNormalDiag(logits, jnp.exp(log_std))
        else:
            pi = distrax.Categorical(logits=logits)

        return pi, activations


class Critic(nn.Module):
    hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        h1 = nn.Dense(self.hidden_size, kernel_init=orthogonal(2.0), bias_init=constant(0.0), name="c_0")(x)
        h1 = act_fn(h1)

        activations = {
            "c_0": None,
            "c_1": h1,
        }
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="c_1",)(h1)

        return value, activations

class ActorCritic(nn.Module):
    action_dim: int
    is_continuous: bool = False
    hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        actor = Actor(self.action_dim, continuous=self.is_continuous, hidden_size=self.hidden_size,
                                activation=self.activation, name='actor')
        pi, actor_activations = actor(obs)

        critic = Critic(hidden_size=self.hidden_size, activation=self.activation, name='critic')

        v, critic_activations = critic(obs)

        return _, pi, jnp.squeeze(v, axis=-1), {"actor": actor_activations, "critic": critic_activations}