import os
import sys

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN.NN.SingleModules.CvT3 import ConvProjectionBlock, CvT3, StageBlock
from pytests.test_equivariance._trasl_equiv_check import traslation_equivariant

key = jax.random.PRNGKey(0)
lattice_size = (8, 8)
C_in = 4
C_out = 8
atol = 1e-10
verbosity = 0


def check(model, channels_in):
    x0 = jax.random.choice(
        key, jnp.array([-1.0, 1.0]), shape=(1, *lattice_size, channels_in)
    )
    params = model.init(key, x0)
    assert traslation_equivariant(x0, params, model, atol=atol, v=verbosity)


def test_ConvProjectionBlock():
    check(ConvProjectionBlock(channels=C_in, n_heads=4), C_in)


def test_StageBlock():
    check(StageBlock(n_CP_blocks=1, channels=C_out, n_heads=4, kernel=(3, 3)), C_in)


def test_StageBlock_strided():
    model = StageBlock(n_CP_blocks=1, channels=C_out, n_heads=4, kernel=(3, 3), strides=(2, 2))
    check(model, C_in)


def test_CvT3():
    model = CvT3(
        n_CP_blocks=(1, 1),
        channels=(2 * C_out, C_out),
        attn_heads=(4, 4),
        kernel=((3, 3), (3, 3)),
        strides=((2, 2), (1, 1)),
        final_architecture=None,
    )
    check(model, 1)


# ------------------------------------------------------------------
# Transformer-block options (activation / ffn / norm / W_O)
# ------------------------------------------------------------------
import pytest

OPTIONS = [
    dict(ffn_act="gelu"),
    dict(ffn_factor=4),
    dict(ffn_out="linear"),
    dict(norm="pre"),
    dict(norm="post"),
    dict(attn_out_proj=True),
    dict(norm="pre", ffn_out="linear", ffn_act="gelu", ffn_factor=4, attn_out_proj=True),
]


@pytest.mark.parametrize("opts", OPTIONS)
def test_CvT3_options_equivariant(opts):
    model = CvT3(
        n_CP_blocks=(1, 1), channels=(C_in, C_out), attn_heads=(2, 2),
        kernel=((3, 3), (3, 3)), strides=((1, 1), (1, 1)), **opts,
    )
    x0 = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(1, *lattice_size, 1))
    params = model.init(key, x0)
    assert traslation_equivariant(x0, params, model, atol=atol, v=verbosity)


def test_CvT3_bad_options():
    x0 = jnp.ones((1, *lattice_size, 1))
    for opts in (dict(norm="xx"), dict(ffn_out="xx")):
        model = CvT3(n_CP_blocks=(1,), channels=(C_in,), attn_heads=(2,),
                     kernel=((3, 3),), strides=((1, 1),), **opts)
        with pytest.raises(ValueError):
            model.init(key, x0)
