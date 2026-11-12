"""
Regression tests for audit bugs in Hydra/NN/NN (Z2, CvTaps APS, Transversal,
CNNPh/SplitTraining, CNN pooling, CNNLiang masks).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_hydra/test_module_fixes.py -q
"""

import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN.NN.FluxModules.SplitTraining import SplitTraining
from Hydra.NN.NN.FluxModules.Transversal import Transversal
from Hydra.NN.NN.SingleModules.CNN import CNN
from Hydra.NN.NN.SingleModules.CNNLiang import CNNLiang
from Hydra.NN.NN.SingleModules.CvTaps import CvTapsWorkerStage
from Hydra.NN.NN.SingleModules.Phase import CNNPh
from Hydra.NN.NN.Symm.Z2 import Z2

L = (4, 4)
N = L[0] * L[1]


def spins(B=8, seed=0):
    return jax.random.choice(jax.random.PRNGKey(seed), jnp.array([-1.0, 1.0]), (B, N))


class RealNet(nn.Module):
    squeeze: bool = False

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(1, param_dtype=jnp.float64)(x)
        return y[:, 0] if self.squeeze else y


class Sign(nn.Module):  # (B,1) of ±1
    @nn.compact
    def __call__(self, x):
        return jnp.sign(x.sum(axis=-1, keepdims=True) + 0.5)


@pytest.mark.parametrize("squeeze", [False, True])
def test_z2_odd_real_and_flat_heads(squeeze):
    x = spins(16)
    m = Z2(Z2Wrap=RealNet(squeeze=squeeze), trivial_Z2=False)
    p = m.init(jax.random.PRNGKey(1), x)
    a, b = m.apply(p, x), m.apply(p, -x)
    assert a.shape == ((16,) if squeeze else (16, 1))
    assert jnp.all(jnp.isfinite(a))
    np.testing.assert_allclose(jnp.exp(b - a), -1.0, atol=1e-10)


def test_cvtaps_strided_stage_translation_invariant():
    x = spins(6).reshape(-1, *L, 1)
    m = CvTapsWorkerStage(n_CP_blocks=1, channels=8, n_heads=2, strides=(2, 2))
    p = m.init(jax.random.PRNGKey(1), x)
    f = lambda y: m.apply(p, y).mean(axis=(1, 2))
    np.testing.assert_allclose(f(jnp.roll(x, 1, axis=1)), f(x), atol=1e-12)


def test_transversal_modulus_marshall_finite():
    x = spins(16)
    m = Transversal(Trans=(RealNet(), Sign()), operation="modulus_marshall")
    p = m.init(jax.random.PRNGKey(1), x)
    out = m.apply(p, x)
    assert jnp.all(jnp.isfinite(out))
    ref = jnp.exp(RealNet().apply({"params": p["params"]["Trans_0"]}, x)) * Sign().apply({}, x)
    np.testing.assert_allclose(jnp.exp(out).ravel(), ref.ravel())


def test_transversal_none_stacks_branches():
    x = spins(4)
    m = Transversal(Trans=(RealNet(), RealNet()), operation="None")
    p = m.init(jax.random.PRNGKey(1), x)
    assert m.apply(p, x).shape == (4, 2, 1)


def test_splittraining_cnnph_shape():
    x = spins(16)
    m = SplitTraining(Modulus=RealNet(), Phase=CNNPh(lattice_size=L, channels=4))
    p = m.init(jax.random.PRNGKey(1), x)
    assert m.apply(p, x).shape == (16, 1)


def test_cnn_use_pooling():
    x = spins(4)
    m = CNN(lattice_size=L, channels=(4,), strides=((1, 1),), use_pooling=True, pooling_strides=((1, 1),), final_architecture=(1,))
    p = m.init(jax.random.PRNGKey(1), x)
    assert m.apply(p, x).shape == (4, 1)


@pytest.mark.parametrize("mask", ["Square", "Triangular"])
def test_cnnliang_mask(mask):
    x = spins(4).reshape(-1, *L, 1)
    m = CNNLiang(lattice_size=L, M1_channels=(6,), M2_channels=(6,), kernel=((3, 3),), mask=mask)
    p = m.init(jax.random.PRNGKey(1), x)
    assert jnp.all(jnp.isfinite(m.apply(p, x)))
