"""CvTsquad = CvT3 with the 'std' block fixed: same parameters, same output, equivariant, f32 switch."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN import REGISTRY_SINGLE  # noqa: E402
from Hydra.NN.NN.SingleModules.CvT3 import CvT3  # noqa: E402
from Hydra.NN.NN.SingleModules.CvTsquad import CvTsquad  # noqa: E402

squad_mod = sys.modules["Hydra.NN.NN.SingleModules.CvTsquad"]  # the package rebinds the name to the class
from Hydra.NN.precision import set_network_dtype  # noqa: E402
from pytests.test_equivariance._trasl_equiv_check import traslation_equivariant  # noqa: E402

key = jax.random.PRNGKey(0)
STD = dict(norm="pre", ffn_out="linear", ffn_act="gelu", ffn_factor=4, attn_out_proj=True)
x0 = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(3, 10, 10, 1))


def test_same_as_cvt3_std():
    squad = CvTsquad()
    params = CvT3(n_CP_blocks=(1, 3), channels=(30, 60), attn_heads=(2, 4),
                  kernel=((3, 3), (3, 3)), strides=((2, 2), (1, 1)), **STD).init(key, x0)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(squad.init(key, x0))
    ref = CvT3(n_CP_blocks=(1, 3), channels=(30, 60), attn_heads=(2, 4),
               kernel=((3, 3), (3, 3)), strides=((2, 2), (1, 1)), **STD).apply(params, x0)
    np.testing.assert_allclose(squad.apply(params, x0), ref, atol=1e-12)


def test_equivariant_and_registered():
    model = CvTsquad(strides=((1, 1), (1, 1)), n_CP_blocks=(1, 1))
    x = x0[:1, :8, :8]
    assert traslation_equivariant(x, model.init(key, x), model, atol=1e-10, v=0)
    assert REGISTRY_SINGLE["CvTsquad"] is CvTsquad


def test_float32_switch():
    try:
        set_network_dtype("float32")
        assert squad_mod.DTYPE == jnp.float32
        out = CvTsquad().init_with_output(key, x0)[0]
        assert out.dtype == jnp.float32
    finally:
        set_network_dtype("float64")
    assert squad_mod.DTYPE == jnp.float64
