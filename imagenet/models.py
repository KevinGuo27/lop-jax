import jax.numpy as jnp
from flax import linen as nn


class ConvNet(nn.Module):
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        x1 = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x1 = nn.relu(x1)
        x1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2))

        x2 = nn.Conv(features=64, kernel_size=(3, 3))(x1)
        x2 = nn.relu(x2)
        x2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2))

        x3 = nn.Conv(features=128, kernel_size=(3, 3))(x2)
        x3 = nn.relu(x3)
        x3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2))
        x3 = x3.reshape((x3.shape[0], -1))  # Flatten

        x4 = nn.Dense(features=128, name='layer_0')(x3)
        x4 = nn.relu(x4)

        x5 = nn.Dense(features=128, name='layer_1')(x4)
        x5 = nn.relu(x5)
        activations = {'layer_0': None,
                       'layer_1': x4,
                       'layer_2': x5}
        x6 = nn.Dense(features=self.num_classes)(x5)
        return x6, activations