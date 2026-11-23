"""
Tensor-network heads next to MPSHead: PEPSHead (backflow PEPS on the token
torus) and SpinMPSHead (MPSHead with one site per spin), arXiv:2603.14425.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_equivariance/test_TNHeads.py -q
"""

import copy
import os
import sys

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
from Hydra.NN.NN.SingleModules.PEPSHead import PEPSHead
from Hydra.NN.NN.SingleModules.SpinMPSHead import SpinMPSHead
from Hydra.NN.NN.SingleModules.ViT2D import ViT2D


def features(shape=(4, 2, 3, 5), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def same_log(a, b, tol=1e-9):
    """Equal complex logs: modulus and phase (mod 2 pi)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.allclose(a.real, b.real, atol=tol) and np.allclose(np.exp(1j * (a.imag - b.imag)), 1, atol=tol)


def layer_norm(p, x):
    mu = x.mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-6)  # flax LayerNorm
    return y * p["scale"] + p["bias"]


def naive_peps_log_psi(params, x, D):
    """Contract the torus PEPS with one einsum over all 2HW bonds, no rescaling."""
    p = params["params"]
    k, b = p["CDense_0"]["ckernel"], p["CDense_0"]["cbias"]
    m = layer_norm(p["LayerNorm_0"], np.asarray(x)) @ (k[..., 0] + 1j * k[..., 1]) + (b[:, 0] + 1j * b[:, 1])
    B, H, W, _ = x.shape
    A = np.asarray(m).reshape(B, H, W, D, D, D, D) + np.einsum("lr,ud->lrud", np.eye(D), np.eye(D))
    letters = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    hb = [[next(letters) for _ in range(W)] for _ in range(H)]  # hb[i][j]: (i, j) -- (i, j+1)
    vb = [[next(letters) for _ in range(W)] for _ in range(H)]  # vb[i][j]: (i, j) -- (i+1, j)
    spec = ",".join(hb[i][j - 1] + hb[i][j] + vb[i - 1][j] + vb[i][j] for i in range(H) for j in range(W))
    return np.log([np.einsum(spec + "->", *[A[s, i, j] for i in range(H) for j in range(W)]) for s in range(B)])


@pytest.mark.parametrize("shape", [(3, 2, 3, 5), (2, 3, 2, 4), (2, 1, 3, 4), (2, 2, 2, 3)])
def test_peps_matches_naive_contraction(shape):
    x = features(shape)
    head = PEPSHead(bond_dim=2, init_scale=1.0)  # far from the identity: the bond structure matters
    p = head.init(jax.random.key(0), x)
    out = head.apply(p, x)
    assert out.shape == (x.shape[0], 1)
    assert same_log(out, naive_peps_log_psi(p, x, 2))


@pytest.mark.parametrize("shift", [(1, 0), (0, 1), (1, 2)])
def test_peps_invariant_under_grid_translations(shift):
    x = features((3, 3, 3, 5))
    head = PEPSHead(bond_dim=2, init_scale=1.0)
    p = head.init(jax.random.key(1), x)
    assert same_log(head.apply(p, jnp.roll(x, shift, axis=(1, 2))), head.apply(p, x))


def test_spin_mps_is_mpshead_on_the_split_tokens():
    """Token (i, j), slot (a, b) of the split goes to spin (i sy + a, j sx + b)."""
    x = features((3, 2, 3, 5))
    sy, sx, h = 2, 3, 4
    head = SpinMPSHead(token_size=(sy, sx), bond_dim=3, features=h, init_scale=1.0)
    p = head.init(jax.random.key(2), x)
    d = p["params"]["Dense_0"]
    y = jax.nn.silu(x @ d["kernel"] + d["bias"]).reshape(3, 2, 3, sy, sx, h)
    grid = np.zeros((3, 2 * sy, 3 * sx, h))
    for i in range(2):
        for j in range(3):
            for a in range(sy):
                for b in range(sx):
                    grid[:, i * sy + a, j * sx + b] = y[:, i, j, a, b]
    ref = MPSHead(bond_dim=3, init_scale=1.0).apply({"params": p["params"]["MPSHead_0"]}, jnp.asarray(grid))
    assert same_log(head.apply(p, x), ref)


@pytest.mark.parametrize("head", [PEPSHead(bond_dim=2), SpinMPSHead(token_size=(2, 2), bond_dim=2)])
def test_needs_feature_grid(head):
    with pytest.raises(ValueError, match="feature grid"):
        head.init(jax.random.key(0), jnp.ones((2, 12, 5)))


def test_peps_runs_under_shard_map():
    """netket evaluates the model inside shard_map."""
    from jax.sharding import Mesh, PartitionSpec as P
    x = features((2, 2, 2, 5))
    head = PEPSHead(bond_dim=2, init_scale=1.0)
    p = head.init(jax.random.key(0), x)
    mesh = Mesh(np.array(jax.devices()[:1]), ("S",))
    f = jax.shard_map(head.apply, mesh=mesh, in_specs=(P(), P("S")), out_specs=P("S"))
    assert same_log(f(p, x), head.apply(p, x))


@pytest.mark.parametrize("head", [PEPSHead(bond_dim=2, init_scale=0.3),
                                  SpinMPSHead(token_size=(2, 2), bond_dim=3, features=4, init_scale=0.3)])
def test_float32_close_to_float64(head):
    x = features((3, 2, 2, 5))
    p = head.init(jax.random.key(2), x)
    ref = head.apply(p, x)
    precision.set_network_dtype("float32")
    try:
        low = head.apply(jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32))
    finally:
        precision.set_network_dtype("float64")
    assert low.dtype == jnp.complex64
    assert same_log(low, ref, tol=1e-4)


def spins(L, n=4, seed=0):
    x = np.tile(np.array([1.0, -1.0] * (L * L // 2)), (n, 1))
    rng = np.random.default_rng(seed)
    for row in x:
        rng.shuffle(row)
    return jnp.asarray(x.reshape(n, L, L, 1))


HEADS = {"PEPS": lambda: PEPSHead(bond_dim=2, init_scale=0.5),
         "SpinMPS": lambda: SpinMPSHead(token_size=(2, 2), bond_dim=3, features=4, init_scale=0.5)}


@pytest.mark.parametrize("name", list(HEADS))
@pytest.mark.parametrize("L, irrep", [(4, (0, 0)), (4, (1, 2)), (6, (0, 0)), (6, (3, 1))])
def test_vit2d_backbone_full_lattice_momentum_and_finite_gradient(name, L, irrep):
    """ViT2D tokens 2x2 + head under Sequivariant with coset: exact momentum of the L x L lattice,
    finite log psi and finite gradient (the 4x4 and 6x6 lattices of the explorer loop)."""
    vit = ViT2D(token_size=(2, 2), embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    m = Sequivariant(SeqV=(vit, HEADS[name]()), irrep=irrep, coset=(2, 2), phase_mode="log")
    x = spins(L)
    p = m.init(jax.random.key(0), x)
    out = m.apply(p, x)
    assert out.shape == (x.shape[0], 1) and np.all(np.isfinite(np.asarray(out)))
    for axis, k in ((1, irrep[0]), (2, irrep[1])):
        d = np.asarray(m.apply(p, jnp.roll(x, 1, axis=axis)) - out)
        assert np.abs(d.real).max() < 1e-9
        np.testing.assert_allclose(np.exp(1j * d.imag), np.exp(2j * np.pi * k / L), atol=1e-9)
    g = jax.grad(lambda q: jnp.sum(m.apply(q, x).real))(p)
    leaves = jax.tree.leaves(g)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in leaves)
    head_key = next(k for k in g["params"] if k.startswith("SeqV_1") or "Head" in k)
    assert any(np.abs(np.asarray(v)).max() > 0 for v in jax.tree.leaves(g["params"][head_key]))


@pytest.mark.parametrize("name, kw", [("PEPSHead", {"bond_dim": 2, "init_scale": 0.5}),
                                      ("SpinMPSHead", {"token_size": [2, 2], "bond_dim": 3, "features": 4})])
@pytest.mark.parametrize("alone", [True, False])
def test_built_by_hydra_from_storage(name, kw, alone):
    """Registered like MPSHead: alone or as Transversal(OutputHead, head), with Z2 and coset (4x4)."""
    from Hydra.NN.Hydra import Hydra
    L = 4
    storage = {"ViT2D": {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1,
                         "n_ffn_layers": 1, "final_architecture": None},
               "OutputHead": {"d_model": 8, "only_phase": False}, name: kw}
    head = {name: {}} if alone else {"Transversal": {"OutputHead": {}, name: {}}}
    config = {"print_arch": False, "load_model": {"parameters": False, "sampler": False}, "storage": storage,
              "symm_wrapper": {"Z2": {"on": True, "trivial_Z2": True, "symm_modulus": False}},
              "selection": "T",
              "T": {"stage0": {"template": {"Sequivariant": {"ViT2D": {}, **head, "irrep": [0, 0],
                                                             "phase_mode": "log", "coset": [2, 2]}}}}}
    hydra = Hydra(copy.deepcopy(config), lattice_size=(L, L), hilbert=nk.hilbert.Spin(0.5, L * L))
    x = spins(L)
    p = hydra.model.init(jax.random.key(0), x.reshape(-1, L * L))
    f = lambda y: hydra.model.apply(p, y.reshape(y.shape[0], L * L))
    ref = f(x)
    assert np.all(np.isfinite(np.asarray(ref)))
    for y in (jnp.roll(x, 1, axis=1), jnp.roll(x, 1, axis=2), -x):
        assert same_log(f(y), ref, tol=1e-8)
