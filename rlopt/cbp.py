from functools import partial

import chex
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp


def top_p_percent_mask(rng, mask, vals, p):
    """
    Calculates the number of eligible elements, then
    the top eligible elements among the eligible elements in vals.
    """
    # Flatten the mask and vals arrays
    mask_flat = mask.flatten()
    vals_flat = vals.flatten()

    # Apply the mask to get values where mask is True
    masked_vals = jnp.where(mask_flat, vals_flat, -jnp.inf)

    # Calculate the number of top elements to select
    ratio = jnp.sum(mask) * (p / 100)
    smaller_than_1 = ratio < 1
    smaller_than_1_sample = jax.random.uniform(rng) <= ratio
    num_top = jnp.floor(ratio).astype(int) + smaller_than_1 * smaller_than_1_sample

    # Get the threshold value for the top p% elements
    threshold = jnp.sort(masked_vals)[-num_top]

    # Create a new mask for elements that are above or equal to the threshold
    top_p_mask_flat = (masked_vals >= threshold) & (mask_flat)

    # Reshape back to the original shape
    return top_p_mask_flat.reshape(mask.shape)


def age_ignoring_top_p_percent_mask(rng, mask, vals, p):
    """
    Separately calculates the number of eligible elements, then
    the top k according to that number in vals.
    """
    # Calculate the number of top elements to select
    ratio = jnp.sum(mask) * (p / 100)
    smaller_than_1 = ratio < 1
    smaller_than_1_sample = jax.random.uniform(rng) <= ratio
    num_top = jnp.floor(ratio).astype(int) + smaller_than_1 * smaller_than_1_sample

    vals_flat = vals.flatten()
    threshold = jnp.sort(vals_flat)[-num_top]

    # Create a new mask for elements that are above or equal to the threshold
    top_p_mask_flat = (vals >= threshold)

    # Reshape back to the original shape
    return top_p_mask_flat.reshape(vals.shape)



def generate_seeds_for_pytree(key, pytree):
    """
    Generate a unique PRNGKey for each leaf in a PyTree.

    Parameters:
    - key: JAX PRNGKey to start generating unique subkeys.
    - pytree: The PyTree for which to generate unique subkeys.

    Returns:
    - A PyTree of the same structure as `pytree`, where each leaf is a unique PRNGKey.
    """
    # Flatten the PyTree and count the leaves
    leaves, treedef = jax.tree.flatten(pytree)

    # Split the original key into as many subkeys as there are leaves
    subkeys = jax.random.split(key, num=len(leaves))

    # Reconstruct the PyTree with the subkeys as leaves
    subkeys_pytree = jax.tree.unflatten(treedef, subkeys)

    return subkeys_pytree


class ContinualBackpropTrainState(TrainState):
    """
    Continual Backprop implementation in JAX.

    We store our utils in a dictionary that is structured much like network_params.
    each key tells you what network it's a part of and the index of the node.
    So a_1 corresponds to the actor, nodes corresponding to layer 1 (layer 0 is the input layer).
    This means that the output weights to layer i, node j are network_params['params']['Actor'][f'a_{i}']['kernel'][j, :]
    which means that the sum for each node should be the sum over axis=-1.
    """
    utils: struct.field(pytree_node=True)
    ages: struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        og_ts = TrainState.create(apply_fn=apply_fn, params=params, tx=tx, **kwargs)

        def is_util_leaf(tree):
            return ('kernel' in tree) and ('bias' in tree)

        def filter_input_and_bias(k, x):
            if k[-1].key[-1] != '0':
                # we do shape[0], since we index layer + 1
                return jnp.zeros(x['kernel'].shape[0])

        # we init our utilities.
        utils = jax.tree_util.tree_map_with_path(filter_input_and_bias, og_ts.params['params'], is_leaf=is_util_leaf)
        ages = jax.tree_util.tree_map_with_path(filter_input_and_bias, og_ts.params['params'], is_leaf=is_util_leaf)

        return ContinualBackpropTrainState(
            step=og_ts.step,
            apply_fn=og_ts.apply_fn,
            params=og_ts.params,
            tx=og_ts.tx,
            opt_state=og_ts.opt_state,
            utils=utils,
            ages=ages,
            **kwargs,
        )

    def update_and_reinit(self,
                          rng: chex.PRNGKey,
                          activations: dict,
                          replacement_rate: float = 1e-4,
                          decay_rate: float = 0.99,
                          maturity_threshold: float = int(1e4)):
        assert jax.tree.flatten(activations)[0][0].shape[0] == 1, 'Not implemented for the batch online setting!'
        activations = jax.tree.map(lambda x: x[0], activations)

        # get_replacement_mask = partial(top_p_percent_mask, p=replacement_rate)
        get_replacement_mask = partial(age_ignoring_top_p_percent_mask, p=replacement_rate)

        # First we update our ages
        new_ages = jax.tree.map(lambda x: x + 1, self.ages)

        # And our utilities
        new_utils = jax.tree.map(lambda x: x * decay_rate, self.utils)
        bias_correction = jax.tree.map(lambda x: 1 - decay_rate ** x, self.ages)

        def get_output_weight_mags(keys, u):
            # This should ignore all input keys.
            if u is None:
                return u

            target_param_set = self.params['params']
            for k in keys[:-1]:
                target_param_set = target_param_set[k.key]

            out_param_set = target_param_set[keys[-1].key]['kernel']

            # Hmmmm paper does a sum instead of a mean here. It should be the same since you're
            # meaning over the same number every time.
            output_weight_mags = jnp.abs(out_param_set).mean(axis=-1)
            return output_weight_mags

        output_weight_mags = jax.tree_util.tree_map_with_path(get_output_weight_mags, new_utils)

        # Calculate our new utils
        u = jax.tree.map(lambda h, w: jnp.abs(h) + w, activations, output_weight_mags)
        new_utils = jax.tree.map(lambda ut, u: decay_rate * ut + (1 - decay_rate) * u, new_utils, u)
        new_bias_corrected_utils = jax.tree.map(lambda nu, bc: nu / bc, new_utils, bias_correction)

        # THIS PART has a discrepancy between the code and the paper.
        # Paper says to replace one feature at a time
        # Now we figure out who CAN update first
        eligiblility_mask = jax.tree.map(lambda age: age > maturity_threshold, new_ages)
        rngs = generate_seeds_for_pytree(rng, eligiblility_mask)

        replacement_mask = jax.tree.map(get_replacement_mask, rngs, eligiblility_mask, new_bias_corrected_utils)

        # Mask our new utils we replace
        new_utils = jax.tree.map(lambda m, u: (1 - m) * u, replacement_mask, new_utils)

        # TODO: zero out optimizer states related to replaced nodes

        # TODO: reinit params

        pass
