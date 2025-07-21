from typing import Any, Callable, List, Optional, Sequence, Tuple
import inspect

import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn


from typing import Any, Callable, List, Optional, Sequence
import inspect

import jax
import jax.numpy as jnp
from flax import linen as nn


class SequentialWithKW(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x, *, feature_list: Optional[List[jnp.ndarray]] = None, train: bool = False):
        for layer in self.layers:
            # forward only the kwargs this layer accepts
            sig = inspect.signature(layer.__call__)
            layer_kwargs = {}
            if 'feature_list' in sig.parameters and feature_list is not None:
                layer_kwargs['feature_list'] = feature_list
            if 'train' in sig.parameters:
                layer_kwargs['train'] = train
            x = layer(x, **layer_kwargs)
        return x


class BasicBlock(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        feature_list: Optional[List[jnp.ndarray]] = None,
        train: bool = False
    ) -> jnp.ndarray:
        identity = x

        # conv1 → bn → relu
        x = nn.Conv(
            self.out_channels, (3, 3), self.stride,
            padding='SAME', use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if feature_list is not None:
            feature_list.append(x)

        # conv2 → bn (no relu)
        x = nn.Conv(
            self.out_channels, (3, 3), (1, 1),
            padding='SAME', use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # downsample identity if needed
        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = nn.Conv(
                self.out_channels, (1, 1), self.stride,
                use_bias=True,
                kernel_init=nn.initializers.kaiming_normal()
            )(identity)
            identity = nn.BatchNorm(use_running_average=not train)(identity)

        # residual add + final relu
        x = x + identity
        x = nn.relu(x)
        if feature_list is not None:
            feature_list.append(x)

        return x


class ResNet18(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        feature_list: Optional[List[jnp.ndarray]] = None,
        train: bool = False
    ) -> tuple[jnp.ndarray, List[jnp.ndarray]]:
        if feature_list is None:
            feature_list = []

        # Initial conv + bn + relu
        x = nn.Conv(
            64, (3, 3), (1, 1),
            padding='SAME', use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        feature_list.append(x)

        # Helper to build a stack of blocks
        def make_layer(in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = []
            layers.append(BasicBlock(in_ch, out_ch, stride))
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch, 1))
            return SequentialWithKW(layers)

        # Four layer groups
        x = make_layer(64,  64,  2, 1)(x, feature_list=feature_list, train=train)
        x = make_layer(64, 128,  2, 2)(x, feature_list=feature_list, train=train)
        x = make_layer(128,256,  2, 2)(x, feature_list=feature_list, train=train)
        x = make_layer(256,512,  2, 2)(x, feature_list=feature_list, train=train)

        # Global average pool + flatten
        x = jnp.mean(x, axis=(1, 2))
        feature_list.append(x)

        # Final classifier
        x = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        return x, feature_list


def build_resnet18(num_classes: int) -> ResNet18:
    return ResNet18(num_classes=num_classes)







# SANITY CHECK
from flax.training import train_state
import optax
from flax.training import common_utils
from flax.training import orbax_utils

class TrainState(train_state.TrainState):
    batch_stats: Any