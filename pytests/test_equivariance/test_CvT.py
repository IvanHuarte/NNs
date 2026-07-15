import os
import sys
from time import time

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.SingleModels.CvT import (
    ConvProjectionBlock,
    CvT,
    DepthPointwiseConv,
    StageBlock,
)
from pytests.test_equivariance._trasl_equiv_check import (
    traslation_equivariant,
)

equivariance_test = traslation_equivariant

key = jax.random.PRNGKey(int(time()))
lattice_size = (8, 8)
C_in = 4
C_out = 8

atol = 1e-10
verbosity = 0


def test_DepthPointwiseConv():
    model = DepthPointwiseConv(channels=C_out)
    x0_shape = (1, lattice_size[0], lattice_size[1], C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_ConvProjectionBlock():

    model = ConvProjectionBlock(channels=C_in, n_heads=4)
    x0_shape = (1, lattice_size[0], lattice_size[1], C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_StageBlock():

    model = StageBlock(
        n_CP_blocks=1,
        channels=C_out,
        n_heads=4,
        kernel=(3, 3),
    )
    x0_shape = (1, lattice_size[0], lattice_size[1], C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_CvT():
    model = CvT(
        lattice_size=lattice_size,
        n_CP_blocks=(2, 2),
        channels=(2 * C_out, C_out),
        attn_heads=(4, 4),
        kernel=(3, 3),
        final_architecture=None,
    )
    x0_shape = (1, lattice_size[0], lattice_size[1], 1)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)
