"""
Depthwise convolution without grouped convs (toolbox.depthwise_conv_circular).

JAX 0.11 cannot take jax.jacrev of a grouped convolution
(feature_group_count > 1) when a global mesh is set, and NetKet sets one at
import: VMC_SR (NTK) failed with NotImplementedError on CvT-like networks.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_depthwise -q
"""

import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket  # noqa: F401  (sets the global mesh that triggers the JAX bug)
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from Hydra.NN.NN.SingleModules.CvT3 import CvT3
from Hydra.NN.NN.SingleModules.CvTaps import ConvAPS
from Hydra.NN.toolbox import (
    DepthPointwiseConv,
    DepthwiseConv,
    depthwise_conv_circular,
)
from Hydra.NN.utils import upgrade_legacy_params

CASES = [
    # (input shape (B,H,W,C), kernel, strides)
    ((2, 4, 4, 5), (3, 3), (1, 1)),
    ((2, 6, 4, 3), (2, 2), (1, 1)),
    ((1, 5, 3, 2), (3, 1), (1, 1)),
    ((1, 6, 6, 4), (5, 5), (1, 1)),
    ((2, 4, 4, 3), (3, 3), (2, 2)),
    ((1, 6, 6, 2), (5, 5), (2, 2)),
    ((1, 5, 7, 2), (3, 3), (2, 3)),
    ((2, 2, 2, 6), (3, 3), (1, 1)),  # kernel larger than the lattice
]


def grouped_conv(shape, kernel_size, strides, use_bias=False, mask=None):
    """Reference: the grouped nn.Conv that is being replaced."""
    C = shape[-1]
    return nn.Conv(
        C,
        kernel_size,
        strides=strides,
        feature_group_count=C,
        padding="CIRCULAR",
        use_bias=use_bias,
        mask=mask,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )


@pytest.mark.parametrize("shape,kernel_size,strides", CASES)
def test_matches_grouped_conv(shape, kernel_size, strides):
    x = jax.random.normal(jax.random.key(0), shape)
    ref = grouped_conv(shape, kernel_size, strides, use_bias=True)
    params = ref.init(jax.random.key(1), x)
    params = jax.tree.map(lambda p: p + 0.3, params)  # non-zero bias too

    mine = DepthwiseConv(kernel_size=kernel_size, strides=strides, use_bias=True)
    # Same parameter names/shapes/init as nn.Conv
    assert jax.tree.map(jnp.shape, mine.init(jax.random.key(1), x)) == jax.tree.map(
        jnp.shape, params
    )
    np.testing.assert_allclose(mine.apply(params, x), ref.apply(params, x), atol=1e-13)


def test_mask():
    shape, ks = (2, 4, 4, 3), (3, 3)
    mask = jnp.zeros((3, 3, 1, 3)).at[1, :].set(1.0)
    x = jax.random.normal(jax.random.key(0), shape)
    ref = grouped_conv(shape, ks, (1, 1), mask=mask)
    params = ref.init(jax.random.key(1), x)
    mine = DepthwiseConv(kernel_size=ks, mask=mask)
    np.testing.assert_allclose(mine.apply(params, x), ref.apply(params, x), atol=1e-13)


def test_jacrev_with_netket_mesh():
    shape = (2, 4, 4, 3)
    x = jax.random.normal(jax.random.key(0), shape)
    ref = grouped_conv(shape, (3, 3), (1, 1))
    params = ref.init(jax.random.key(1), x)

    with pytest.raises(NotImplementedError):  # the JAX limitation being avoided
        jax.jacrev(lambda p: ref.apply(p, x).ravel()[:3])(params)

    kernel = params["params"]["kernel"]
    jac = jax.jacrev(lambda k: depthwise_conv_circular(x, k).ravel()[:3])(kernel)
    jac_fwd = jax.jacfwd(lambda k: ref.apply({"params": {"kernel": k}}, x).ravel()[:3])(kernel)
    np.testing.assert_allclose(jac, jac_fwd, atol=1e-13)


def test_depth_pointwise_conv_tree_and_dtypes():
    x = jnp.ones((1, 4, 4, 6))
    params = DepthPointwiseConv(channels=8).init(jax.random.key(0), x)["params"]
    assert set(params) == {"Conv_0", "LayerNorm_0", "Conv_1"}
    assert params["Conv_0"]["kernel"].shape == (3, 3, 1, 6)
    assert params["Conv_1"]["kernel"].shape == (1, 1, 6, 8)
    assert all(p.dtype == jnp.float64 for p in jax.tree.leaves(params))


def test_conv_aps_depthwise_path():
    """CvTaps' ConvAPS (strided, circular, depthwise) without grouped convs."""
    shape, ks, strides = (2, 4, 4, 3), (3, 3), (2, 2)
    x = jax.random.normal(jax.random.key(0), shape)
    aps = ConvAPS(
        features=3, kernel_size=ks, strides=strides, feature_group_count=3, padding="CIRCULAR"
    )
    params = aps.init(jax.random.key(1), x)
    assert set(params["params"]) == {"Conv_0"}

    ref = nn.Conv(
        3, ks, strides=strides, feature_group_count=3, padding="CIRCULAR",
        use_bias=False, dtype=jnp.float64, param_dtype=jnp.float64,
    )
    from Hydra.NN.NN.SingleModules.CvTaps import APS_equivariance_adapter

    expected = ref.apply(
        {"params": params["params"]["Conv_0"]}, APS_equivariance_adapter(x, strides)
    )
    np.testing.assert_allclose(aps.apply(params, x), expected, atol=1e-13)
    jax.jacrev(lambda p: aps.apply(p, x).ravel()[:2])(params)  # no NotImplementedError


def test_cvt3_float64_and_jacrev():
    model = CvT3(
        n_CP_blocks=(1, 1),
        channels=(6, 8),
        attn_heads=(2, 2),
        kernel=((3, 3), (3, 3)),
        strides=((2, 2), (1, 1)),
        final_architecture=(4, 1),
    )
    x = jax.random.normal(jax.random.key(0), (2, 4, 4, 1))
    params = model.init(jax.random.key(1), x)
    assert all(p.dtype == jnp.float64 for p in jax.tree.leaves(params))
    jax.jacrev(lambda p: model.apply(p, x).ravel())(params)


def test_legacy_single_precision_upcast():
    legacy = {
        "params": {
            "SymmModel": {
                "c": np.ones(2, np.complex64),
                "a": np.ones(3, np.float32),
                "b": np.ones(2, np.float64),
            }
        }
    }
    out = upgrade_legacy_params(legacy)["params"]["NN"]
    assert list(out) == ["c", "a", "b"]  # order kept: Hydra codes depend on it
    assert out["a"].dtype == np.float64
    assert out["b"].dtype == np.float64
    assert out["c"].dtype == np.complex128
