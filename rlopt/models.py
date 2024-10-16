import distrax
import flax.linen as nn
from jax._src.nn.initializers import orthogonal, constant
import jax.numpy as jnp


class Actor(nn.Module):
    action_dim: int
    continuous: bool = False
    activation: str = 'tanh'
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        activation = nn.relu
        if self.activation == 'tanh':
            activation = nn.tanh

        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if self.continuous:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)
        return pi


class Critic(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return critic


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
        return out




class ActorCritic(nn.Module):
    action_dim: int
    is_continuous: bool = False
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        embedding = SimpleNN(hidden_size=self.hidden_size)(x)

        actor = Actor(self.action_dim, continuous=self.is_continuous, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        v = critic(embedding)

        return pi, jnp.squeeze(v, axis=-1)
