#!/usr/bin/env python3
"""
Process all experiment bundles and run dead neuron analysis on each one.
This creates individual dormant unit analysis files for each bundle/seed.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import orbax.checkpoint as ocp

sys.path.append(str(Path(__file__).parent.parent))
from incremental_cifar.model import build_resnet18

def load_cifar_from_pickle():
    with open('/users/tserapio/lop-jax/incremental_cifar/data/cifar100-onehot.pkl', "rb") as f:
        import pickle
        data = pickle.load(f)
    x_train = np.asarray(data["x_train"]) # expected NCHW
    y_train = np.asarray(data["y_train"]) # should be one-hot
    return x_train, y_train

def one_batch_for_classes(
    x: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    _cls = np.asarray(classes, dtype=np.int32)
    y_ids = np.argmax(y, axis=1)
    mask = np.isin(y_ids, _cls)
    x_sel = x[mask]
    if x_sel.size == 0:
        raise RuntimeError(f"No samples found for classes {_cls.tolist()}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_sel), size=min(batch_size, len(x_sel)), replace=False)
    xb = x_sel[idx]
    # NCHW -> NHWC
    if xb.ndim == 4 and xb.shape[1] in (1, 3):
        xb = np.transpose(xb, (0, 2, 3, 1))
    return xb.astype(np.float32)

def extract_params_and_batch_stats(state):
    """Extract (params, batch_stats) from a TrainState-like dict/object."""
    if isinstance(state, dict) and "params" in state:
        params = state["params"]
        batch_stats = state.get("batch_stats", {})
        return params, batch_stats
    
    if hasattr(state, "params"):
        params = state.params
        batch_stats = getattr(state, "batch_stats", {})
        return params, batch_stats
    
    raise ValueError("Could not find params/batch_stats in train state.")

def run_forward_collect_features(
    model: nn.Module,
    params: dict,
    batch_stats: dict,
    images: np.ndarray,
) -> dict:
    """Runs a forward pass and returns the model's feature_dict."""
    variables = {"params": params}
    if batch_stats:
        variables["batch_stats"] = batch_stats
    feats = {}
    logits, feature_dict = model.apply(
        variables,
        images,
        feature_dict=feats,
        train=False,
    )
    return feature_dict

def compute_dormant_all_layers(
    feature_dict: dict,
    threshold: float,
) -> float:
    """Compute dormant proportion across all layers in feature_dict."""
    total_units = 0
    dormant = 0

    def count_layer(x: jnp.ndarray):
        if x.ndim == 4:  # NHWC conv
            act_freq = (x != 0).astype(jnp.float32).mean(axis=(0, 1, 2))
            return int(act_freq.shape[0]), int((act_freq < threshold).sum())
        if x.ndim == 2:  # (N, D) dense
            act_freq = (x != 0).astype(jnp.float32).mean(axis=0)
            return int(act_freq.shape[0]), int((act_freq < threshold).sum())
        return 0, 0

    vals = list(feature_dict.values())[-2:]  # last two layers ONLY
    for v in vals:
        if v is None:
            continue
        if isinstance(v, dict):
            for bv in v.values():
                if bv is None:
                    continue
                n, d = count_layer(jnp.asarray(bv))
                total_units += n
                dormant += d
        else:
            n, d = count_layer(jnp.asarray(v))
            total_units += n
            dormant += d
    
    return dormant

def align_tree_to_reference(restored, reference):
    """Function to remove leading 1's in shape to match reference."""
    def _align(r, ref):
        r = np.asarray(r)
        ref = np.asarray(ref)
        while r.ndim > ref.ndim:
            r = r[0]
        return r
    return jax.tree.map(_align, restored, reference)

def find_all_bundles(results_base_dir):
    """Find all experiment bundles across all agents."""
    results_base = Path(results_base_dir)
    
    all_bundles = []
    
    # Loop through agent directories
    for agent_dir in results_base.iterdir():
        if not agent_dir.is_dir():
            continue
            
        agent_name = agent_dir.name
        
        # Find experiment bundles in this agent directory
        experiment_dirs = [
            d for d in agent_dir.iterdir() 
            if d.is_dir() and d.name.startswith("incremental_cifar_seed")
        ]
        
        for exp_dir in experiment_dirs:
            # Extract seed from directory name
            try:
                seed = exp_dir.name.split("(")[1].split(")")[0]
            except:
                seed = "unknown"
            
            all_bundles.append({
                'agent': agent_name,
                'bundle_path': exp_dir,
                'seed': seed,
                'output_name': f"{agent_name}_seed{seed}"
            })
    
    return all_bundles

def run_dead_neuron_analysis(bundle_info, base_results_dir, args):
    """Run dead neuron analysis on a single bundle."""
    
    bundle_path = bundle_info['bundle_path'].resolve()
    print(f"\nProcessing bundle: {bundle_path}")
    agent = bundle_info['agent']
    output_name = bundle_info['output_name']
    
    # Create output directory for this specific bundle
    output_dir = Path(base_results_dir) / args.agents[0] / "dormant_analysis" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {output_name}: {bundle_path}")
    
    try:
        # Load the bundle
        ckptr = ocp.PyTreeCheckpointer()
        bundle = ckptr.restore(bundle_path)
        
        if not (isinstance(bundle, dict) and "train_states_per_task" in bundle):
            print(f"✗ Bundle does not contain 'train_states_per_task': {output_name}")
            return False

        train_states_by_task = bundle["train_states_per_task"]
        num_steps = len(train_states_by_task)
        if num_steps == 0:
            print(f"✗ train_states_per_task is empty: {output_name}")
            return False

        # Get class order from bundle if available, otherwise use default
        if "class_order" in bundle:
            class_order = bundle["class_order"]
            if isinstance(class_order, (list, tuple, np.ndarray)):
                class_order = np.asarray(class_order).flatten()
                print(f"  Using class order from bundle: {class_order.tolist()}")
        else:
            class_order = np.asarray([92, 53, 76, 9, 50, 37, 57, 65, 23, 61, 14, 89, 39, 96, 78, 30, 63, 60,  1, 10, 95, 45, 43, 18, 33, 32, 88, 41, 72, 73, 36, 11, 94, 82, 21, 69, 75, 62,  5, 90, 52, 51, 34, 13, 83, 20, 97, 56, 71, 35, 79,  8, 68,  0, 74, 12,  6, 19, 86, 58, 24, 26, 27, 80,  2, 25, 29, 70, 15, 42, 67, 64, 77, 54, 40, 99, 66, 17, 47, 46, 93, 22, 48, 28,  4, 49, 87,  7, 31, 91, 81, 55, 59, 98, 84, 85,  3, 38, 16, 44])

        # Load CIFAR data
        x_train, y_train = load_cifar_from_pickle()

        # Build model
        model = build_resnet18(num_classes=100)

        # Results arrays
        dorm_before = np.zeros(num_steps, dtype=np.float32)
        dorm_after = np.zeros(num_steps, dtype=np.float32)

        for i, ts in enumerate(train_states_by_task):
            try:
                params, batch_stats = extract_params_and_batch_stats(ts)
                dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
                vars_ref = None
                if args.seed and (type(args.seed) is int):
                    vars_ref = model.init(jax.random.PRNGKey(args.seed), dummy, feature_dict={}, train=False)
                else:
                    vars_ref = model.init(jax.random.PRNGKey(2027), dummy, feature_dict={}, train=False)
                params_ref = vars_ref['params']
                batch_stats_ref = vars_ref.get('batch_stats', {})
                params = align_tree_to_reference(params, params_ref)
                batch_stats = align_tree_to_reference(batch_stats, batch_stats_ref)

                # NEXT task classes
                next_cls = class_order[(i * args.classes_per_task):((i + 1) * args.classes_per_task)]
                xb_next = one_batch_for_classes(x_train, y_train, next_cls, args.batch_size, seed=args.seed + i)
                feat_next = run_forward_collect_features(model, params, batch_stats, xb_next)
                dorm_after[i] = compute_dormant_all_layers(feat_next, args.dormant_unit_threshold)

                # PREVIOUS tasks (if i > 0)
                if i > 0:
                    prev_cls = class_order[: (i * args.classes_per_task)]
                    xb_prev = one_batch_for_classes(x_train, y_train, prev_cls, args.batch_size, seed=args.seed + i)
                    feat_prev = run_forward_collect_features(model, params, batch_stats, xb_prev)
                    dorm_before[i] = compute_dormant_all_layers(feat_prev, args.dormant_unit_threshold)

            except Exception as e:
                print(f"Warning: Error processing task {i}: {e}")
                continue

        # Save results
        prev_path = output_dir / "previous_tasks_dormant_units_analysis.npy"
        next_path = output_dir / "next_task_dormant_units_analysis.npy"

        np.save(prev_path, dorm_before)
        np.save(next_path, dorm_after)
        
        print(f"Successfully processed {output_name}")
        return True
        
    except Exception as e:
        print(f"Error processing {output_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process all bundles for dead neuron analysis")
    parser.add_argument("--results_base", type=str, 
                       default="/users/tserapio/lop-jax/incremental_cifar/results",
                       help="Base results directory containing agent subdirectories")
    parser.add_argument("--classes_per_task", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dormant_unit_threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", nargs="+", default=None,
                       help="Specific agents to process (default: all)")
    
    args = parser.parse_args()
    
    print(f"Finding all bundles in {args.results_base}")
    all_bundles = find_all_bundles(args.results_base)
    
    # Filter by agents if specified
    if args.agents:
        all_bundles = [b for b in all_bundles if b['agent'] in args.agents]
    
    if not all_bundles:
        print("No bundles found!")
        return
    
    print(f"Found {len(all_bundles)} bundles to process:")
    for bundle in all_bundles:
        print(f"  {bundle['output_name']}: {bundle['bundle_path']}")
    
    # Process each bundle
    successful = 0
    failed = 0
    
    for bundle_info in tqdm(all_bundles, desc="Processing bundles"):
        if run_dead_neuron_analysis(bundle_info, args.results_base, args):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved in: {args.results_base}/{args.agents[0]}/dormant_analysis/")

if __name__ == "__main__":
    main()
