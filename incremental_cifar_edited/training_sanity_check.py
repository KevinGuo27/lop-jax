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

# scroll down for sanity check

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
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1), padding='VALID')
        x = x.reshape((x.shape[0], -1))  # flatten to (batch, features)
        feature_list.append(x)

        # Final classifier - no linear in linen, use Dense
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

def make_lr_scheduler(num_tasks: int,
                        base_lr: float,
                        base_steps_per_epoch: int,
                        epochs_per_task: int,
                        drop_factor: float = 0.2,
                        drop_epochs: tuple = (60, 120, 160)):

        # at what steps do we have a new task? 
        task_starts = [0]
        for t in range(1, num_tasks):
            S = base_steps_per_epoch * t
            task_starts.append(task_starts[-1] + S * epochs_per_task)

        def lr_fn(global_step: int) -> float:
            # which task are we on?
            t = next(i for i, start in enumerate(task_starts)
                 if i == len(task_starts)-1 or global_step < task_starts[i+1])
            
            # which epoch are we on?
            # local step within the task
            S = base_steps_per_epoch * (t+1)
            local_step = global_step - task_starts[t]
            local_epoch = local_step // S

            # piecewise constant lr schedule within the task
            if local_epoch < drop_epochs[0]:
                return base_lr 
            elif local_epoch < drop_epochs[1]:
                return base_lr * drop_factor
            elif local_epoch < drop_epochs[2]:
                return base_lr * (drop_factor**2)
            else:
                return base_lr * (drop_factor**3)

        return lr_fn

lr_schedule = make_lr_scheduler(
    num_tasks = 20, #20 for incremental cifar-100
    base_lr = 0.1, #0.1 for incremental cifar-100
    base_steps_per_epoch = 25,
    epochs_per_task = 200,
    drop_factor = 0.2,
    drop_epochs = (60, 120, 160),
)

# TRAINING SANITY CHECK

model = build_resnet18(num_classes=100)
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)), train=True)
print(variables['batch_stats'].keys())
state = TrainState.create(
    apply_fn=model.apply, 
    params=variables['params'], 
    batch_stats=variables['batch_stats'],
    tx=optax.chain(
        optax.add_decayed_weights(1e-5), 
        optax.sgd(learning_rate=lr_schedule, momentum=0.9, nesterov=False)
    )
)

dummy_inputs = jnp.ones((2, 32, 32, 3))
dummy_labels = jnp.array([0, 1])
(output_full, features), updates = state.apply_fn(variables, x=dummy_inputs, train=True, mutable='batch_stats')
output = output_full[:, :2]  # only first 2 classes for this dummy example
loss_fn = lambda params: 1.0 # dummy loss
grads = jax.grad(loss_fn)(state.params)
state = state.apply_gradients(grads=grads)
state = state.replace(batch_stats=updates['batch_stats'])
print(variables['batch_stats'].keys())

print(output[0])

accuracy = jnp.mean(jnp.argmax(output, axis=-1) == dummy_labels)
print(f"Output shape: {output.shape}, Accuracy: {accuracy*100:.2f}%")
for i, feat in enumerate(features):
    print(f"Feature shape: {feat.shape}")