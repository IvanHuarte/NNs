import os
import sys

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN.NN.SingleModules.CNNLiang import CNNLiangBlock
from pytests.test_equivariance._trasl_equiv_check import traslation_equivariant

key = jax.random.PRNGKey(0)
lattice_size = (4, 4)
C_in = 16
atol = 1e-10
verbosity = 0


def make_block(window_pooling):
    model = CNNLiangBlock(
        M1_channels=C_in,
        M2_channels=2 * C_in,
        kernel=(3, 3),
        window_pooling=window_pooling,
        use_bias=False,
        mask=None,
    )
    x0 = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(1, *lattice_size, C_in))
    return model, x0, model.init(key, x0)


def test_CNNLiangBlock_without_pooling_is_fully_equivariant():
    model, x0, params = make_block(window_pooling=1)
    assert traslation_equivariant(x0, params, model, atol=atol, v=verbosity)


def test_CNNLiangBlock_pooling_equivariant_subgroup():
    """Max-pooling acts on the flattened (H*W) lattice, pairing consecutive
    sites of each row: the block is equivariant under any vertical shift and
    horizontal shifts multiple of window_pooling (not under odd ones)."""
    model, x0, params = make_block(window_pooling=2)
    y0 = model.apply(params, x0)
    for dy in range(lattice_size[0]):
        for dx in range(0, lattice_size[1], 2):
            y = model.apply(params, jnp.roll(x0, (dy, dx), axis=(1, 2)))
            assert jnp.allclose(jnp.roll(y0, (dy, dx), axis=(1, 2)), y, atol=atol), (dy, dx)
