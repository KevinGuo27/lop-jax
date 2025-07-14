import time
import os
import pickle
from copy import deepcopy
import json
import argparse
from functools import partialmethod

from tqdm import tqdm
import numpy as np

# jax imports
from flax import nnx
from functools import partial
import jax
import jax.numpy as jnp
import optax

import torch
from torch.utils.data import DataLoader # still use pytorch dataloader

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
from mlproj_manager.file_management.file_and_directory_management import store_object_with_several_attempts

# updated imports with jax implementations
# from modified_resnet_nnx import build_resnet18
from modified_resnet_nnx_cbp import build_resnet18
from res_gnt_jax import ResGnT

# for hessian computation at the start and end of each task 
from utils.hessian_computation import get_hvp_fn
from utils.lanczos import lanczos_alg
from utils.density import tridiag_to_density
from utils.optimizer import l2_regularization, adam_with_param_counts
from utils.file_system import get_results_path, numpyify, plot_hessian_spectrum

class Model(nnx.Module):
  def __init__(self, rngs):
    self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    self.bn = nnx.BatchNorm(4, rngs=rngs)
  def __call__(self, x):
    return self.bn(self.linear2(self.linear1(x)))

loss_fn = lambda model: ((model(x) - y) ** 2).mean()
def update(opt):
  loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
  opt.update(grads) # in-place update
  return opt, loss

def dummy(opt):
  print(opt.model(x))

x = jax.random.normal(jax.random.key(0), (1, 2))
y = jnp.ones((1, 4))

model = Model(nnx.Rngs(0))
tx = optax.adam(1e-3)
state = nnx.Optimizer(model, tx)

model.train()
dummy(state)

model.eval()
dummy(state)