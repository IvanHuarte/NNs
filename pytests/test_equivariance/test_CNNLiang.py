import os
import sys
from time import time

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.SingleModels.CNNLiang import ConvBlock
from pytests.test_equivariance._trasl_equiv_check import (
    traslation_equivariant,
)

equivariance_test = traslation_equivariant
atol = 1e-10
verbosity = 1


key = jax.random.PRNGKey(int(time()))
lattice_size = (4, 4)
M1 = 16
M2 = 32
kernel = (3, 3)
window_pooling = 2
use_bias = False
mask = None


def test_ConvBlock():
    model = ConvBlock(
        M1_channels=M1,
        M2_channels=M2,
        kernel=kernel,
        window_pooling=window_pooling,
        use_bias=use_bias,
        mask=mask,
    )
    x0_shape = (1, *lattice_size, M1)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)
