import distrax
import flax.linen as nn
from jax._src.nn.initializers import orthogonal, constant
import jax.numpy as jnp


class Actor(nn.Module):
    action_dim: int
    continuous: bool = False
    activation: str = 'relu'
    h_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        activation = nn.relu
        if self.activation == 'tanh':
            activation = nn.tanh


        # TODO: lecun init?
        out_0 = nn.Dense(self.h_dims[0], kernel_init=orthogonal(2), bias_init=constant(0.0),
                         name='a_0')(
            x
        )
        out_0 = activation(out_0)
        activations = {'a_0': out_0}

        out_i = out_0

        for i in range(1, len(self.h_dims)):
            out_i = nn.Dense(self.h_dims[i], kernel_init=orthogonal(2), bias_init=constant(0.0),
                             name=f'a_{i}')(
                out_i
            )
            out_i = activation(out_i)
            activations[f'a_{i}'] = out_i

        out_i_plus_1 = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name=f'a_{len(self.h_dims)}_output'
        )(out_i)

        if self.continuous:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(out_i_plus_1, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=out_i_plus_1)
        return pi, activations


class Critic(nn.Module):
    h_dims: tuple = (256, 256)
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x):
        activation = nn.relu
        if self.activation == 'tanh':
            activation = nn.tanh

        # TODO: lecun init?
        out_0 = nn.Dense(self.h_dims[0], kernel_init=orthogonal(2), bias_init=constant(0.0),
                         name='c_0')(
            x
        )
        out_0 = activation(out_0)
        activations = {'c_0': out_0}

        out_i = out_0

        for i in range(1, len(self.h_dims)):
            out_i = nn.Dense(self.h_dims[i], kernel_init=orthogonal(2), bias_init=constant(0.0),
                             name=f'c_{i}')(
                out_i
            )
            out_i = activation(out_i)
            activations[f'c_{i}'] = out_i

        out_i_plus_1 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name=f'c_{len(self.h_dims)}_output')(
            out_i
        )
        return out_i_plus_1, activations


# class SimpleNN(nn.Module):
#     hidden_size: int
#
#     @nn.compact
#     def __call__(self, x):
#         out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
#             x
#         )
#         out = nn.relu(out)
#         out = nn.Dense(
#             self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(out)
#         out = nn.relu(out)
#         out = nn.Dense(
#             self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(out)
#         return out


class ActorCritic(nn.Module):
    action_dim: int
    is_continuous: bool = False
    h_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):

        actor = Actor(self.action_dim, continuous=self.is_continuous, h_dims=self.h_dims,
                      name='actor')
        pi, actor_activations = actor(x)

        critic = Critic(h_dims=self.h_dims, name='critic')
        v, critic_activations = critic(x)

        return pi, jnp.squeeze(v, axis=-1), {'actor': actor_activations, 'critic': critic_activations}
