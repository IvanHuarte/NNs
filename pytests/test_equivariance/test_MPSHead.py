"""
MPSHead: backflow-MPS readout (arXiv:2603.14425) of a backbone feature grid.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_equivariance/test_MPSHead.py -q
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
from Hydra.NN import REGISTRY, precision
from Hydra.NN.Hydra import Hydra
from Hydra.NN.NN.FluxModules.Sequivariant import Sequivariant
from Hydra.NN.NN.SingleModules.ViT2D import ViT2D

MPSHead = REGISTRY["MPSHead"]


def features(shape=(6, 3, 4, 5), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def naive_log_psi(params, x, D, n):
    """log sum_j sum_c Tr prod_n A^j along the row-major path shifted by c, no rescaling."""
    p = params["params"]
    mu = x.mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-6)  # flax LayerNorm
    y = y * p["LayerNorm_0"]["scale"] + p["LayerNorm_0"]["bias"]
    k, b = p["CDense_0"]["ckernel"], p["CDense_0"]["cbias"]
    m = y @ (k[..., 0] + 1j * k[..., 1]) + (b[:, 0] + 1j * b[:, 1])
    B, H, W, _ = x.shape
    A = np.asarray(m).reshape(B, H, W, n, D, D) + np.eye(D)
    out = np.zeros(B, complex)
    for bi in range(B):
        for j in range(n):
            for c in range(W):
                M = np.eye(D)
                for i in range(H):
                    for col in range(W):
                        M = M @ A[bi, i, (col + c) % W, j]
                out[bi] += np.trace(M)
    return np.log(out)


def same_log(a, b, tol=1e-9):
    """Equal complex logs: modulus and phase (mod 2 pi)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.allclose(a.real, b.real, atol=tol) and np.allclose(np.exp(1j * (a.imag - b.imag)), 1, atol=tol)


def test_matches_naive_contraction():
    x = features()
    head = MPSHead(bond_dim=3, n_mps=2, init_scale=1.0)  # A far from 1: the order matters
    p = head.init(jax.random.key(0), x)
    out = head.apply(p, x)
    assert out.shape == (x.shape[0], 1)
    assert same_log(out, naive_log_psi(p, x, 3, 2))


@pytest.mark.parametrize("shift", [(1, 0), (0, 1), (2, 3)])
def test_invariant_under_grid_translations(shift):
    x = features()
    head = MPSHead(bond_dim=3, init_scale=1.0)
    p = head.init(jax.random.key(1), x)
    assert same_log(head.apply(p, jnp.roll(x, shift, axis=(1, 2))), head.apply(p, x))


def test_runs_under_shard_map():
    """netket evaluates the model inside shard_map: a constant scan carry is rejected there."""
    from jax.sharding import Mesh, PartitionSpec as P
    x = features()
    head = MPSHead(bond_dim=3, init_scale=1.0)
    p = head.init(jax.random.key(0), x)
    mesh = Mesh(np.array(jax.devices()[:1]), ("S",))
    f = jax.shard_map(head.apply, mesh=mesh, in_specs=(P(), P("S")), out_specs=P("S"))
    assert same_log(f(p, x), head.apply(p, x))


def test_needs_feature_grid():
    head = MPSHead(bond_dim=2)
    with pytest.raises(ValueError, match="feature grid"):
        head.init(jax.random.key(0), jnp.ones((2, 12, 5)))


L = 8


def spins(n=5):
    x = np.tile(np.array([1.0, -1.0] * (L * L // 2)), (n, 1))
    rng = np.random.default_rng(0)
    for row in x:
        rng.shuffle(row)
    return jnp.asarray(x.reshape(n, L, L, 1))


@pytest.mark.parametrize("irrep", [(0, 0), (3, 1)])
def test_tokens_plus_coset_give_full_lattice_momentum(irrep):
    """Stride/token 2x2: the head sees a 4x4 grid, the coset adds the intra-token shifts."""
    vit = ViT2D(token_size=(2, 2), embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    m = Sequivariant(SeqV=(vit, MPSHead(bond_dim=3, init_scale=0.3)), irrep=irrep,
                     coset=(2, 2), phase_mode="log")
    x = spins()
    p = m.init(jax.random.key(0), x)
    for axis, k in ((1, irrep[0]), (2, irrep[1])):
        d = np.asarray(m.apply(p, jnp.roll(x, 1, axis=axis)) - m.apply(p, x))
        assert np.abs(d.real).max() < 1e-9
        np.testing.assert_allclose(np.exp(1j * d.imag), np.exp(2j * np.pi * k / L), atol=1e-9)


def test_float32_close_to_float64():
    x = features()
    head = MPSHead(bond_dim=3, init_scale=0.3)
    p = head.init(jax.random.key(2), x)
    ref = head.apply(p, x)
    precision.set_network_dtype("float32")
    try:
        low = head.apply(jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32))
    finally:
        precision.set_network_dtype("float64")
    assert low.dtype == jnp.complex64
    assert same_log(low, ref, tol=1e-4)


def test_transversal_branch_keeps_a_trained_outputhead():
    """Warm start: OutputHead weights -> Trans_0, MPSHead at small init_scale changes log psi little."""
    storage = {
        "ViT2D": {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1,
                  "n_ffn_layers": 1, "final_architecture": None},
        "OutputHead": {"d_model": 8, "only_phase": False},
        "MPSHead": {"bond_dim": 4, "n_mps": 2, "init_scale": 1e-3},
    }
    seqv = {"irrep": [0, 0], "phase_mode": "log", "coset": [2, 2]}
    config = {
        "print_arch": False,
        "load_model": {"parameters": False, "sampler": False},
        "storage": storage,
        "symm_wrapper": {"Z2": {"on": False, "trivial_Z2": True, "symm_modulus": False}},
        "selection": "T",
        "T": {
            "stage0": {"template": {"Sequivariant": {"ViT2D": {}, "OutputHead": {}, **seqv}},
                       "save_params": True},
            "stage1": {"template": {"Sequivariant": {
                           "ViT2D": {}, "Transversal": {"OutputHead": {}, "MPSHead": {}}, **seqv}},
                       "load": [["stage0", "0", "0"], ["stage0", "1", "10"]]},
        },
    }
    hydra = Hydra(copy.deepcopy(config), lattice_size=(L, L), hilbert=nk.hilbert.Spin(0.5, L * L))
    p0 = hydra.model.init(jax.random.key(0), jnp.ones((1, L * L)))["params"]
    x = spins(16).reshape(16, L * L)
    l0 = hydra.model.apply({"params": p0}, x)

    hydra.arch_evol(p0)
    p1 = hydra.weight_transplantation(hydra.model.init(jax.random.key(7), jnp.ones((1, L * L)))["params"])
    assert "Trans_1" in p1["NN"]["SeqV_1"] and "CDense_0" in p1["NN"]["SeqV_1"]["Trans_1"]
    d = np.asarray(hydra.model.apply({"params": p1}, x) - l0)
    assert np.std(d.real) < 1e-2 and np.std(d.imag) < 1e-2
