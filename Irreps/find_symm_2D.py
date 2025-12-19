#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

#  %%
import enum

import jax
import jax.nn
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.typing
import netket as nk
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 30

# %%
N_A = 3
N_B = 3
JS = [1.0, 0.5]


class Z2_Irrep(enum.Enum):
    TRIVIAL = 0
    SIGN = 1
