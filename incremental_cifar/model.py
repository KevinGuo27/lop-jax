from typing import Any, Callable, List, Optional, Sequence, Tuple
import inspect

import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn


from typing import Any, Callable, List, Optional, Sequence
import inspect

KERNEL_INIT_RELU = nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
KERNEL_INIT_LINEAR = nn.initializers.variance_scaling(1.0, "fan_out", "truncated_normal")
BIAS_INIT_ZEROS = nn.initializers.zeros

class SequentialWithKW(nn.Module):
    """Minimal Sequential that forwards only kwargs a child block can accept."""

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        feature_dict: Optional[dict] = None,
        train: bool = False,
        id: Optional[str] = None,
    ) -> jnp.ndarray:
        for idx, layer in enumerate(self.layers):
            layer_id = f"{id}/block{idx}" if id is not None else None
            sig = inspect.signature(layer.__call__)
            kw = {}
            if "feature_dict" in sig.parameters and feature_dict is not None:
                kw["feature_dict"] = feature_dict
            if "train" in sig.parameters:
                kw["train"] = train
            if "id" in sig.parameters and layer_id is not None:
                kw["id"] = layer_id
            x = layer(x, **kw)
        return x


class BasicBlock(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1
    zero_init_residual: bool = False

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

        x1 = nn.Conv(
            self.out_channels, 
            (3, 3), 
            self.stride,
            padding='SAME', 
            use_bias=True,
            kernel_init=KERNEL_INIT_RELU, 
            bias_init=BIAS_INIT_ZEROS,
            name='Conv_0'
        )(x)
        x1 = nn.BatchNorm(use_running_average=not train, momentum=0.9, name='BatchNorm_0')(x1)
        x1 = nn.relu(x1)

        x2 = nn.Conv(
            self.out_channels, 
            (3, 3), 
            (1, 1),
            padding='SAME', use_bias=True,
            kernel_init=KERNEL_INIT_RELU,
            bias_init=BIAS_INIT_ZEROS,
            name='Conv_1'
        )(x1)
        x2 = nn.BatchNorm(
            use_running_average=not train, 
            momentum=0.9, 
            scale_init=nn.initializers.zeros if self.zero_init_residual else nn.initializers.ones,
            name='BatchNorm_1'
        )(x2)

        # downsample identity if needed
        if self.stride != 1 or self.in_channels != self.out_channels:
            init_identity = identity
            identity = nn.Conv(
                self.out_channels, 
                (1, 1), 
                self.stride,
                use_bias=True,
                kernel_init=KERNEL_INIT_RELU,
                bias_init=BIAS_INIT_ZEROS,
                name='Identity'
            )(identity)
            identity = nn.BatchNorm(use_running_average=not train, momentum=0.9, name='BatchNorm_2')(identity)
            x3 = x2 + identity
            x3 = nn.relu(x3)
            if feature_dict is not None and id is not None:
                feature_dict[f'{self.name}'] = {'BatchNorm_0': None, 'BatchNorm_1': None, 'BatchNorm_2': None, 'Conv_0': None, 'Conv_1': x1, 'Identity': init_identity}
            return x3


        # residual add + final relu
        x3 = x2 + identity
        x3 = nn.relu(x3)
        if feature_dict is not None and id is not None:
            feature_dict[f'{self.name}'] = {'BatchNorm_0': None, 'BatchNorm_1': None, 'Conv_0': None, 'Conv_1': x2}

        return x3


class ResNet18(nn.Module):
    num_classes: int
    zero_init_residual: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        feature_dict: Optional[dict] = None,
        train: bool = False
    ) -> tuple[jnp.ndarray, dict]:
        if feature_dict is None:
            feature_dict = {}

        # Initial conv + bn + relu
        x = nn.Conv(
            64, 
            (3, 3), 
            (1, 1),
            padding='SAME', 
            use_bias=True,
            kernel_init=KERNEL_INIT_RELU,
            bias_init=BIAS_INIT_ZEROS,
            name='Conv_0'
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name='BatchNorm_0')(x)
        x = nn.relu(x)
        feature_dict['BatchNorm_0'] = None
        feature_dict['Conv_0'] = None

        # Helper to build a stack of blocks
        def make_layer(in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = []
            layers.append(BasicBlock(in_ch, out_ch, stride, zero_init_residual=self.zero_init_residual))
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch, 1, zero_init_residual=self.zero_init_residual))
            return SequentialWithKW(layers)

        # Four layer groups
        x = make_layer(64,  64,  2, 1)(x, feature_dict=feature_dict, train=train, id='layer1')
        x = make_layer(64, 128,  2, 2)(x, feature_dict=feature_dict, train=train, id='layer2')
        x = make_layer(128,256,  2, 2)(x, feature_dict=feature_dict, train=train, id='layer3')
        x = make_layer(256,512,  2, 2)(x, feature_dict=feature_dict, train=train, id='layer4')

        # Global average pool + flatten
        x = jnp.mean(x, axis=(1, 2))

        # Final classifier - no linear in linen, use Dense
        x = nn.Dense(
            512,
            use_bias=True,
            kernel_init=KERNEL_INIT_LINEAR, 
            bias_init=BIAS_INIT_ZEROS,
            name='Dense_0'
        )(x)
        x1 = nn.relu(x)
        x2 = nn.Dense(
            512,
            use_bias=True,
            kernel_init=KERNEL_INIT_LINEAR, 
            bias_init=BIAS_INIT_ZEROS,
            name='Dense_1'
        )(x1)
        x2 = nn.relu(x2)
        feature_dict['Dense_0'] = None
        feature_dict['Dense_1'] = x1
        feature_dict['Dense_2'] = x2
        x3 = nn.Dense(
            self.num_classes, 
            use_bias=True,
            kernel_init=KERNEL_INIT_LINEAR, 
            bias_init=BIAS_INIT_ZEROS,
            name='Dense_2'
        )(x2)
        return x3, feature_dict


def build_resnet18(num_classes: int) -> ResNet18:
    return ResNet18(num_classes=num_classes)