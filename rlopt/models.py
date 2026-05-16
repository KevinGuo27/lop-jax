import distrax
import jax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant, lecun_uniform
import numpy as np
import functools

class Actor(nn.Module):
    action_dim: int
    continuous: bool = False
    hidden_size: int = 256
    activation: str = "relu"
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        h1 = nn.Dense(self.hidden_size, kernel_init=lecun_uniform(), bias_init=constant(0.0), name="a_0")(x)
        if self.use_layernorm:
            h1 = nn.LayerNorm(name="a_ln_0")(h1)
        h1 = act_fn(h1)

        activations = {
            "a_0": None,
            "a_1": h1,
        }

        logits = nn.Dense(self.action_dim, kernel_init=lecun_uniform(), bias_init=constant(0.0), name="a_1")(h1)

        if self.continuous:
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            activations["log_std"] = None
            pi = distrax.MultivariateNormalDiag(logits, jnp.exp(log_std))
        else:
            pi = distrax.Categorical(logits=logits)

        return pi, activations


class Critic(nn.Module):
    hidden_size: int = 256
    activation: str = "relu"
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        h1 = nn.Dense(self.hidden_size, kernel_init=lecun_uniform(), bias_init=constant(0.0), name="c_0")(x)
        if self.use_layernorm:
            h1 = nn.LayerNorm(name="c_ln_0")(h1)
        h1 = act_fn(h1)

        activations = {
            "c_0": None,
            "c_1": h1,
        }
        value = nn.Dense(1, kernel_init=lecun_uniform(), bias_init=constant(0.0), name="c_1")(h1)

        return value, activations

class ActorCritic(nn.Module):
    action_dim: int
    is_continuous: bool = False
    hidden_size: int = 256
    activation: str = "relu"
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x):
        obs = x

        actor = Actor(
            self.action_dim,
            continuous=self.is_continuous,
            hidden_size=self.hidden_size,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name='actor'
        )
        pi, actor_activations = actor(obs)

        critic = Critic(
            hidden_size=self.hidden_size,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name='critic'
        )

        v, critic_activations = critic(obs)

        return pi, jnp.squeeze(v, axis=-1), {"actor": actor_activations, "critic": critic_activations}