import jax
import jax.numpy as jnp
from time import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.CvTaps import ConvAPS
from pytests.test_equivariance_APS._APS_equiv_check import APS_equiv_check

equivariance_test = APS_equiv_check

key = jax.random.PRNGKey(int(time()))

lattice_size = (8, 8)
strides = (2, 2)

C_in = 6
C_out = 6

atol = 1e-3
verbosity = 1


def test_Conv_APS():
    model = ConvAPS(
        channels=C_out,
        strides=strides,
        padding="CIRCULAR",
    )
    x0_shape = (1, *lattice_size, C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, params, model, atol=atol, v=verbosity)
