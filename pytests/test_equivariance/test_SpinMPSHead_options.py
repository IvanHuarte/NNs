"""
SpinMPSHead options real / marshall / physical (arXiv:2603.14425) and the ViT2DSpins backbone.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_equivariance/test_SpinMPSHead_options.py -q
"""

import copy
import importlib
import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN import precision
from Hydra.NN.NN.FluxModules.Sequivariant import Sequivariant
from Hydra.NN.NN.SingleModules.MPSHead import MPSHead
from Hydra.NN.NN.SingleModules.SpinMPSHead import SpinMPSHead, mps_chain
from Hydra.NN.NN.SingleModules.ViT2D import Tokenize, ViT2D
from Hydra.NN.NN.SingleModules.ViT2DSpins import ViT2DSpins
from Hydra.NN.toolbox import MarshallSign

SM = importlib.import_module("Hydra.NN.NN.SingleModules.SpinMPSHead")  # the package attribute is the class


class FrozenSpinMPSHead(nn.Module):
    """SpinMPSHead before the options (22 nov 2026): reference for the default path."""

    token_size: tuple[int, int]
    bond_dim: int
    features: int = 16
    n_mps: int = 2
    init_scale: float = 0.5
    init_type: str = "identity"

    @nn.compact
    def __call__(self, x):
        B, H, W, _ = x.shape
        sy, sx = self.token_size
        h = self.features
        y = nn.Dense(sy * sx * h, dtype=SM.DTYPE, param_dtype=SM.DTYPE)(x.astype(SM.DTYPE))
        y = nn.silu(y).reshape(B, H, W, sy, sx, h).transpose(0, 1, 3, 2, 4, 5).reshape(B, H * sy, W * sx, h)
        return MPSHead(bond_dim=self.bond_dim, n_mps=self.n_mps, init_scale=self.init_scale, init_type=self.init_type)(y)


def same_log(a, b, tol=1e-9):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.allclose(a.real, b.real, atol=tol) and np.allclose(np.exp(1j * (a.imag - b.imag)), 1, atol=tol)


def rand_spins(B, L0, L1, seed=0):
    return np.random.default_rng(seed).choice([-1.0, 1.0], (B, L0, L1))


def head_input(B=4, H=3, W=3, C=6, tok=(2, 2), seed=0):
    """(features, token-grid input with the spins appended as ViT2DSpins does, spin lattice)."""
    feats = jnp.asarray(np.random.default_rng(seed).standard_normal((B, H, W, C)))
    s = rand_spins(B, H * tok[0], W * tok[1], seed + 1)
    tokens = Tokenize(tok, (H, W), tok[0] * tok[1])(jnp.asarray(s))
    return feats, jnp.concatenate([feats, tokens], -1), s


@pytest.mark.parametrize("init_type", ["identity", "nilpotent"])
@pytest.mark.parametrize("low", [False, True])
def test_defaults_identical_to_frozen_copy(init_type, low):
    x = jnp.asarray(np.random.default_rng(0).standard_normal((5, 3, 3, 12)))
    kw = dict(token_size=(2, 2), bond_dim=4, features=6, init_scale=0.3, init_type=init_type)
    new, old = SpinMPSHead(**kw), FrozenSpinMPSHead(**kw)
    p = new.init(jax.random.key(1), x)
    q = old.init(jax.random.key(1), x)
    assert jax.tree.structure(p) == jax.tree.structure(q)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(q)))
    if low:
        precision.set_network_dtype("float32")
        p, x = jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32)
    try:
        assert np.array_equal(new.apply(p, x), old.apply(p, x))
        g_new = jax.grad(lambda t: jnp.sum(jnp.abs(new.apply(t, x)) ** 2))(p)
        g_old = jax.grad(lambda t: jnp.sum(jnp.abs(old.apply(t, x)) ** 2))(p)
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(g_new), jax.tree.leaves(g_old)))
    finally:
        precision.set_network_dtype("float64")


def test_vit2dspins_is_vit2d_plus_token_spins():
    kw = dict(token_size=(2, 2), embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    s = rand_spins(3, 6, 6)
    x = jnp.asarray(s[..., None])
    p = ViT2D(**kw).init(jax.random.key(0), x)
    assert jax.tree.structure(p) == jax.tree.structure(ViT2DSpins(**kw).init(jax.random.key(0), x))
    f, g = ViT2D(**kw).apply(p, x), ViT2DSpins(**kw).apply(p, x)
    assert g.shape == (3, 3, 3, 12) and np.array_equal(g[..., :8], f)
    for i in range(3):
        for j in range(3):
            assert np.array_equal(g[:, i, j, 8:], s[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(3, 4))


def test_marshall_multiplies_by_the_marshall_sign():
    feats, x, s = head_input()
    kw = dict(token_size=(2, 2), bond_dim=3, features=5, init_scale=0.5)
    head = SpinMPSHead(marshall=True, **kw)
    p = head.init(jax.random.key(0), x)
    d = np.asarray(head.apply(p, x) - SpinMPSHead(**kw).apply(p, feats)).ravel()  # same parameters
    sign = np.asarray(MarshallSign(lattice_size=s.shape[1:]).apply({}, jnp.asarray(s.reshape(len(s), -1)))).ravel()
    assert np.abs(d.real).max() == 0
    np.testing.assert_allclose(np.exp(1j * d.imag), np.exp(1j * sign), atol=1e-12)


def test_physical_contracts_the_spin():
    """A_n = 1 + T[s_n].y_n with T[s] = T1 + s T2: MPSHead on [LN(y), s LN(y)]."""
    feats, x, s = head_input()
    head = SpinMPSHead(token_size=(2, 2), bond_dim=3, features=5, init_scale=0.5, physical=True)
    p = head.init(jax.random.key(0), x)["params"]
    d, ln = p["Dense_0"], p["LayerNorm_0"]
    y = jax.nn.silu(feats @ d["kernel"] + d["bias"]).reshape(4, 3, 3, 2, 2, 5).transpose(0, 1, 3, 2, 4, 5)
    y = y.reshape(4, 6, 6, 5)
    mu = y.mean(-1, keepdims=True)
    y = (y - mu) / jnp.sqrt(((y - mu) ** 2).mean(-1, keepdims=True) + 1e-6) * ln["scale"] + ln["bias"]
    ref = MPSHead(bond_dim=3, init_scale=0.5).apply({"params": p["MPSHead_0"]},
                                                    jnp.concatenate([y, s[..., None] * y], -1))
    assert same_log(head.apply({"params": p}, x), ref, tol=1e-12)
    flipped = x.at[:, 0, 0, -1].multiply(-1)  # one spin, same features
    assert not same_log(head.apply({"params": p}, flipped), head.apply({"params": p}, x), tol=1e-6)


def naive_chain(mats):
    B, H, W, n = mats.shape[:4]
    out = np.zeros(B, complex)
    for b in range(B):
        for j in range(n):
            for c in range(W):
                M = np.eye(mats.shape[-1])
                for i in range(H):
                    for col in range(W):
                        M = M @ mats[b, i, (col + c) % W, j]
                out[b] += np.trace(M)
    return np.log(out.astype(complex))


@pytest.mark.parametrize("cplx", [False, True])
def test_chain_matches_naive_contraction(cplx):
    rng = np.random.default_rng(3)
    mats = rng.standard_normal((3, 2, 3, 2, 3, 3)) + (1j * rng.standard_normal((3, 2, 3, 2, 3, 3)) if cplx else 0)
    assert same_log(mps_chain(jnp.asarray(mats)), naive_chain(mats))


def test_real_amplitude_is_real():
    _, x, _ = head_input()
    for kw in (dict(real=True), dict(real=True, marshall=True, physical=True)):
        head = SpinMPSHead(token_size=(2, 2), bond_dim=3, features=5, init_scale=1.0, **kw)
        out = np.asarray(head.apply(head.init(jax.random.key(0), x), x))
        assert np.abs(np.sin(out.imag)).max() < 1e-12
    assert "CDense_0" not in str(jax.tree_util.tree_structure(head.init(jax.random.key(0), x)))


OPTIONS = [dict(real=True), dict(marshall=True), dict(physical=True), dict(real=True, marshall=True, physical=True)]


@pytest.mark.parametrize("opts", OPTIONS, ids=lambda o: "+".join(o))
@pytest.mark.parametrize("L, irrep", [(4, (0, 0)), (4, (1, 2)), (6, (0, 0)), (6, (3, 1))])
def test_options_full_lattice_momentum_and_finite_gradient(opts, L, irrep):
    vit = ViT2DSpins(token_size=(2, 2), embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    head = SpinMPSHead(token_size=(2, 2), bond_dim=3, features=4, init_scale=0.3, **opts)
    m = Sequivariant(SeqV=(vit, head), irrep=irrep, coset=(2, 2), phase_mode="log")
    x = jnp.asarray(rand_spins(4, L, L)[..., None])
    p = m.init(jax.random.key(0), x)
    out = m.apply(p, x)
    assert np.all(np.isfinite(np.asarray(out)))
    for axis, k in ((1, irrep[0]), (2, irrep[1])):
        d = np.asarray(m.apply(p, jnp.roll(x, 1, axis=axis)) - out)
        assert np.abs(d.real).max() < 1e-9
        np.testing.assert_allclose(np.exp(1j * d.imag), np.exp(2j * np.pi * k / L), atol=1e-9)
    g = jax.grad(lambda q: jnp.sum(m.apply(q, x).real))(p)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(g))


@pytest.mark.parametrize("opts", [dict(marshall=True), dict(real=True, marshall=True, physical=True)],
                         ids=lambda o: "+".join(o))
def test_built_by_hydra_with_z2(opts):
    """ViT2DSpins + SpinMPSHead through the registry, with Z2 and coset (4x4): translations and spin flip."""
    from Hydra.NN.Hydra import Hydra
    L = 4
    storage = {"ViT2DSpins": {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1,
                              "n_ffn_layers": 1, "final_architecture": None},
               "SpinMPSHead": {"token_size": [2, 2], "bond_dim": 3, "features": 4, "init_scale": 0.3, **opts}}
    config = {"print_arch": False, "load_model": {"parameters": False, "sampler": False}, "storage": storage,
              "symm_wrapper": {"Z2": {"on": True, "trivial_Z2": True, "symm_modulus": False}},
              "selection": "T",
              "T": {"stage0": {"template": {"Sequivariant": {"ViT2DSpins": {}, "SpinMPSHead": {}, "irrep": [0, 0],
                                                             "phase_mode": "log", "coset": [2, 2]}}}}}
    hydra = Hydra(copy.deepcopy(config), lattice_size=(L, L), hilbert=nk.hilbert.Spin(0.5, L * L))
    x = jnp.asarray(rand_spins(4, L, L)[..., None])
    p = hydra.model.init(jax.random.key(0), x.reshape(-1, L * L))
    f = lambda y: hydra.model.apply(p, y.reshape(y.shape[0], L * L))
    ref = f(x)
    assert np.all(np.isfinite(np.asarray(ref)))
    for y in (jnp.roll(x, 1, axis=1), jnp.roll(x, 1, axis=2), -x):
        assert same_log(f(y), ref, tol=1e-8)


def test_options_float32_close_to_float64():
    _, x, _ = head_input()
    head = SpinMPSHead(token_size=(2, 2), bond_dim=3, features=5, init_scale=0.3, real=True, marshall=True, physical=True)
    p = head.init(jax.random.key(2), x)
    ref = head.apply(p, x)
    precision.set_network_dtype("float32")
    try:
        low = head.apply(jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32))
    finally:
        precision.set_network_dtype("float64")
    assert low.dtype == jnp.complex64
    assert same_log(low, ref, tol=1e-4)
