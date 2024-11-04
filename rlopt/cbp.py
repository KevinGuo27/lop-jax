from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp


class ContinualBackpropTrainState(TrainState):
    utils: struct.field(pytree_node=True)
    ages: struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        og_ts = TrainState.create(apply_fn=apply_fn, params=params, tx=tx, **kwargs)

        def filter_output_and_bias(k, x):
            if 'output' not in k[-2].key and k[-1] != 'bias':
                # we do shape[-1], since this is the out_features shape
                return jnp.zeros(x.shape[-1])
        # we init our utilities.
        # output layers have names that end in '_output'
        utils = jax.tree_util.tree_map_with_path(filter_output_and_bias, og_ts.params['params'])
        ages = jax.tree_util.tree_map_with_path(filter_output_and_bias, og_ts.params['params'])

        print()

    def update_utils(self):
        pass

    def shake_and_bake(self, activations, params,
                       replacement_rate: float = 1e-4,
                       decay_rate: float = 0.99,
                       maturity_threshold: float = int(1e4)):
        pass