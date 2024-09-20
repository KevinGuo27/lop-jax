import hashlib
from pathlib import Path
import time

from rlopt.config import Hyperparams

from definitions import ROOT_DIR


def get_results_path(args: Hyperparams, return_npy: bool = True):
    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)

    args_hash = make_hash_md5(args.as_dict())
    time_str = time.strftime("%Y%m%d-%H%M%S")

    if args.study_name is not None:
        results_dir /= args.study_name
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.id_str()}_time({time_str})_{args_hash}{'.npy' if return_npy else ''}"
    return results_path


def make_hash_md5(o):
    return hashlib.md5(str(o).encode('utf-8')).hexdigest()


