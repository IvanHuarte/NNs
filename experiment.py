#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

import jax

print("JAX version:", jax.__version__)

# Backend y dispositivos
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

# Información adicional de GPU:
for dev in jax.devices():
    print("-----")
    print("Device:", dev)
    print("Platform:", dev.platform)
    print("Device kind:", dev.device_kind)

import sys

sys.exit()

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
import jaxlib
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

from typing import Tuple, Callable, Any, Dict, Optional
import numpy.typing as npt
import copy
import pathlib
import matplotlib.pyplot as plt
import time
import json
import ast
import netket as nk
import os
import glob
import sys

sys.path.append("/home/ihuarte/Escritorio/Ivan/NNs")

# os.chdir("/home/ihuarte/Escritorio/Ivan/NN/")

from VA_project.model.model import J1J2Square
from VA_project.engine.runners import Runner

# from NN_utils import load_vstate
# from correlations import correlations_vstate

# from NNs.NN_module.ST_utils import compare_params, masked_optimizer
from frozendict import deepfreeze
from NN_module.ST_utils import print_tree


# %%

configurations = [
    "/home/ihuarte/Escritorio/Ivan/NNs/config_Hydra.json",
    "/home/ihuarte/Escritorio/Ivan/NNs/config_Hydra_NN.json",
]
with open(configurations[0], "r") as f:
    config_hydra = json.load(f)
with open(configurations[1], "r") as f:
    config_nn = json.load(f)

config = {**config_hydra, **config_nn}

storage = config["storage"]
symm_wrappers = config["symm_wrappers"]
arch_evolution = config[config["selection"]]

from NN_module.NN.Hydra import Hydra
from NN_module.NN import __all_single__

hydra = Hydra(config, **{"lattice_size": [4, 4]})
# %%

hydra.update_info()
