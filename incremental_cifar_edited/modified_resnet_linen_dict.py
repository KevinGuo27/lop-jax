from typing import Any, Callable, List, Optional, Sequence, Tuple
import inspect

import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn


from typing import Any, Callable, List, Optional, Sequence
import inspect


class SequentialWithKW(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x, *, feature_dict: Optional[dict] = None, train: bool = False, id: Optional[str] = None):
        for layer in self.layers:
            # forward only the kwargs this layer accepts
            sig = inspect.signature(layer.__call__)
            layer_kwargs = {}
            if 'feature_dict' in sig.parameters and feature_dict is not None:
                layer_kwargs['feature_dict'] = feature_dict
            if 'train' in sig.parameters:
                layer_kwargs['train'] = train
            if 'id' in sig.parameters and id is not None:
                layer_kwargs['id'] = f'{id}/block{id}'
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
        feature_dict: Optional[dict] = None,
        train: bool = False, 
        id: Optional[str] = None
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
        if feature_dict is not None and id is not None:
            feature_dict[f'{id}/layer'] = x

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
        if feature_dict is not None and id is not None:
            feature_dict[f'{id}/layer'] = x

        return x


class ResNet18(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        feature_dict: Optional[dict] = None,
        train: bool = False
    ) -> tuple[jnp.ndarray, List[jnp.ndarray]]:
        if feature_dict is None:
            feature_dict = {}

        # Initial conv + bn + relu
        x = nn.Conv(
            64, (3, 3), (1, 1),
            padding='SAME', use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        feature_dict['conv1'] = x

        # Helper to build a stack of blocks
        def make_layer(in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = []
            layers.append(BasicBlock(in_ch, out_ch, stride))
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch, 1))
            return SequentialWithKW(layers)

        # Four layer groups
        x = make_layer(64,  64,  2, 1)(x, feature_dict=feature_dict, train=train, id='layer1')
        x = make_layer(64, 128,  2, 2)(x, feature_dict=feature_dict, train=train, id='layer2')
        x = make_layer(128,256,  2, 2)(x, feature_dict=feature_dict, train=train, id='layer3')
        x = make_layer(256,512,  2, 2)(x, feature_dict=feature_dict, train=train, id='layer4')

        # Global average pool + flatten
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1), padding='VALID')
        x = x.reshape((x.shape[0], -1))  # flatten to (batch, features)
        feature_dict['avgpool'] = x

        # Final classifier - no linear in linen, use Dense
        x = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        return x, feature_dict


def build_resnet18(num_classes: int) -> ResNet18:
    return ResNet18(num_classes=num_classes)