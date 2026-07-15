import os
import sys
from time import time

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.SingleModels.CvT import (
    ConvProjectionBlock,
    CvTWorker,
    DepthPointwiseConv,
    StageBlock,
)
from pytests.test_equivariance._trasl_equiv_check import (
    traslation_equivariant,
)

equivariance_test = traslation_equivariant

key = jax.random.PRNGKey(int(time()))
lattice_size = (4, 4)
C_in = 16
C_out = 32

atol = 1e-10
verbosity = 0


def test_DepthPointwiseConv():
    model = DepthPointwiseConv(channels=C_out)
    x0_shape = (1, *lattice_size, C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_ConvProjectionBlock():

    model = ConvProjectionBlock(channels=C_in, n_heads=4)
    x0_shape = (1, *lattice_size, C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_StageBlock():

    model = StageBlock(
        n_CP_blocks=1, CTemb_channels=C_out, CP_channels=C_out, n_heads=4
    )
    x0_shape = (1, *lattice_size, C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_CvTWorker():

    model = CvTWorker(
        lattice_size=lattice_size,
        n_CP_blocks_list=(2, 2),
        CTemb_channels_list=(2 * C_out, C_out),
        CP_channels_list=(2 * C_out, C_out),
        attn_heads_list=(2, 4),
    )
    x0_shape = (1, lattice_size[0], lattice_size[1], 1)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)
