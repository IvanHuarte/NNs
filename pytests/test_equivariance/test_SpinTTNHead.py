"""
SpinTTNHead: backflow tree tensor network with one leaf per spin (explorer papers.md section 5).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_equivariance/test_SpinTTNHead.py -q
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
from Hydra.NN.NN.SingleModules.SpinTTNHead import SpinTTNHead, ttn_log_psi
from Hydra.NN.NN.SingleModules.ViT2D import ViT2D


def same_log(a, b, tol=1e-9):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.allclose(a.real, b.real, atol=tol) and np.allclose(np.exp(1j * (a.imag - b.imag)), 1, atol=tol)


def rand(shape, cplx, rng, scale=1.0):
    a = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
    return scale * a


def tree(H, W, K=4, dims=(2, 3, 3), B=2, cplx=True, seed=0):
    rng = np.random.default_rng(seed)
    d0, d1, d2 = dims
    return (rand((B, H, W, K, d0), cplx, rng), rand((d1, d0**K), cplx, rng), rand((d2, d1**W), cplx, rng),
            rand((1, d2**H), cplx, rng))


def naive_tree(leaves, w1, w2, w3):
    """psi = sum over the H*W grid shifts of the tree, each shift one einsum over every bond (no rescaling)."""
    B, H, W, K, d0 = leaves.shape
    d1, d2 = w1.shape[0], w2.shape[0]
    letters = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    Lf = [[[next(letters) for _ in range(K)] for _ in range(W)] for _ in range(H)]
    T = [[next(letters) for _ in range(W)] for _ in range(H)]
    R = [next(letters) for _ in range(H)]
    out = np.zeros(B, complex)
    for b in range(B):
        for c in range(W):
            for s in range(H):
                specs = ["".join(R[(l + s) % H] for l in range(H))]
                ops = [w3.reshape((d2,) * H)]
                for rho in range(H):
                    specs.append(R[rho] + "".join(T[rho][(j + c) % W] for j in range(W)))
                    ops.append(w2.reshape((d2,) + (d1,) * W))
                for i in range(H):
                    for j in range(W):
                        specs.append(T[i][j] + "".join(Lf[i][j]))
                        ops.append(w1.reshape((d1,) + (d0,) * K))
                        for k in range(K):
                            specs.append(Lf[i][j][k])
                            ops.append(leaves[b, i, j, k])
                out[b] += np.einsum(",".join(specs) + "->", *ops, optimize="greedy")
    return np.log(out.astype(complex))


@pytest.mark.parametrize("cplx", [True, False])
@pytest.mark.parametrize("H, W", [(2, 2), (3, 3), (2, 3), (1, 3)])
def test_tree_matches_brute_force(H, W, cplx):
    """4x4 (2x2 token grid), 6x6 (3x3), and non-square grids."""
    args = tree(H, W, cplx=cplx)
    assert same_log(ttn_log_psi(*map(jnp.asarray, args)), naive_tree(*args))


def leaves_of(head, p, x):
    """The leaf vectors of the module, recomputed from its parameters."""
    q = p["params"]
    B, H, W, _ = x.shape
    K, h = head.token_size[0] * head.token_size[1], head.features
    y = jax.nn.silu(x @ q["Dense_0"]["kernel"] + q["Dense_0"]["bias"]).reshape(B, H, W, K, h)
    mu = y.mean(-1, keepdims=True)
    y = (y - mu) / jnp.sqrt(((y - mu) ** 2).mean(-1, keepdims=True) + 1e-6)
    y = y * q["LayerNorm_0"]["scale"] + q["LayerNorm_0"]["bias"]
    if head.real:
        return y @ q["Dense_1"]["kernel"] + q["Dense_1"]["bias"]
    k, b = q["CDense_0"]["ckernel"], q["CDense_0"]["cbias"]
    return y @ (k[..., 0] + 1j * k[..., 1]) + (b[:, 0] + 1j * b[:, 1])


def nodes_of(p, real):
    q = p["params"]
    return [q[n] if real else q[n][..., 0] + 1j * q[n][..., 1] for n in ("w_token", "w_row", "w_top")]


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("g", [2, 3])
def test_module_is_the_tree_of_its_leaves(g, real):
    x = jnp.asarray(np.random.default_rng(1).standard_normal((3, g, g, 7)))
    head = SpinTTNHead(token_size=(2, 2), leaf_dim=3, token_dim=4, row_dim=4, features=5, init_scale=0.7, real=real)
    p = head.init(jax.random.key(0), x)
    out = head.apply(p, x)
    assert out.shape == (3, 1) and out.dtype == jnp.complex128
    assert same_log(out, ttn_log_psi(leaves_of(head, p, x), *nodes_of(p, real)), tol=1e-12)
    if real:
        assert np.abs(np.sin(np.asarray(out.imag))).max() < 1e-12  # real psi: phase 0 or pi


@pytest.mark.parametrize("low", [False, True])
def test_log_scale_stability(low):
    """Multilinear: leaf n times e^a_n -> log psi + sum a_n; node times l -> l^(number of copies). No overflow."""
    H = W = 3
    leaves, w1, w2, w3 = map(jnp.asarray, tree(H, W, seed=3))
    a = jnp.asarray(np.random.default_rng(4).uniform(-1, 1, leaves.shape[:4])) * (30 if low else 300)
    base = ttn_log_psi(leaves, w1, w2, w3)
    cases = [((leaves * jnp.exp(a)[..., None], w1, w2, w3), a.sum((1, 2, 3))[:, None]),
             ((leaves, w1 * 1e-6, w2 * 1e8, w3 * 1e-30), np.log(1e-6) * H * W + np.log(1e8) * H + np.log(1e-30))]
    for args, shift in cases:
        if low:
            args = [t.astype(jnp.complex64) for t in args]
        out = ttn_log_psi(*args)
        assert np.all(np.isfinite(np.asarray(out)))
        assert same_log(np.asarray(out, np.complex128) - shift, base, tol=2e-3 if low else 1e-8)


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("init_scale", [0.0, 0.05, 1.0])
def test_finite_gradients_at_init_and_at_random_parameters(init_scale, real):
    x = jnp.asarray(np.random.default_rng(2).standard_normal((4, 3, 3, 6)))
    head = SpinTTNHead(token_size=(2, 2), init_scale=init_scale, real=real)
    p = head.init(jax.random.key(0), x)
    noisy = jax.tree.map(lambda v: v + 0.5 * jax.random.normal(jax.random.key(9), v.shape, v.dtype), p)
    for q in (p, noisy):
        for f in (lambda t: jnp.sum(head.apply(t, x).real), lambda t: jnp.sum(head.apply(t, x).imag)):
            g = jax.grad(f)(q)
            assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(g))
    g = jax.grad(lambda t: jnp.sum(head.apply(t, x).real))(p)  # flat start is not a dead point
    assert np.abs(np.asarray(g["params"]["CDense_0" if not real else "Dense_1"]["ckernel" if not real else "kernel"])).max() > 0


def test_flat_log_psi_at_zero_init_scale():
    x = jnp.asarray(np.random.default_rng(5).standard_normal((16, 3, 3, 8)))
    head = SpinTTNHead(token_size=(2, 2), init_scale=0.0)
    out = np.asarray(head.apply(head.init(jax.random.key(0), x), x))
    assert np.ptp(out.real) < 1e-12 and np.ptp(out.imag) < 1e-12


def test_jit_vmap_and_lax_map_over_chunks():
    x = jnp.asarray(np.random.default_rng(6).standard_normal((12, 3, 3, 6)))
    head = SpinTTNHead(token_size=(2, 2), init_scale=0.5)
    p = head.init(jax.random.key(0), x)
    ref = head.apply(p, x)
    f = lambda c: head.apply(p, c)
    chunks = x.reshape(3, 4, 3, 3, 6)
    for out in (jax.jit(head.apply)(p, x), jax.vmap(f)(chunks).reshape(12, 1), jax.lax.map(f, chunks).reshape(12, 1)):
        assert same_log(out, ref, tol=1e-12)


def spins(L, n=4, seed=0):
    x = np.tile(np.array([1.0, -1.0] * (L * L // 2)), (n, 1))
    return jnp.asarray(np.random.default_rng(seed).permuted(x, axis=1).reshape(n, L, L, 1))


@pytest.mark.parametrize("L, irrep", [(4, (0, 0)), (4, (1, 2)), (6, (3, 1))])
def test_sequivariant_coset_characters(L, irrep):
    """Shifts by one site (coset) and by one token give exp(2 pi i k.t / L)."""
    vit = ViT2D(token_size=(2, 2), embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    m = Sequivariant(SeqV=(vit, SpinTTNHead(token_size=(2, 2), init_scale=0.5)), irrep=irrep, coset=(2, 2),
                     phase_mode="log")
    x = spins(L)
    p = m.init(jax.random.key(0), x)
    out = m.apply(p, x)
    for axis, k in ((1, irrep[0]), (2, irrep[1])):
        for t in (1, 2):
            d = np.asarray(m.apply(p, jnp.roll(x, t, axis=axis)) - out)
            assert np.abs(d.real).max() < 1e-9
            np.testing.assert_allclose(np.exp(1j * d.imag), np.exp(2j * np.pi * k * t / L), atol=1e-9)


@pytest.mark.parametrize("even", [True, False])
def test_built_by_hydra_with_z2_on_4x4(even):
    """Registry + storage (token_size as a list), coset (2,2), Z2 even/odd: translations and the spin flip."""
    from Hydra.NN.Hydra import Hydra
    L = 4
    storage = {"ViT2D": {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1, "n_ffn_layers": 1,
                         "final_architecture": None},
               "SpinTTNHead": {"token_size": [2, 2], "leaf_dim": 4, "token_dim": 8, "row_dim": 8, "init_scale": 0.5}}
    config = {"print_arch": False, "load_model": {"parameters": False, "sampler": False}, "storage": storage,
              "symm_wrapper": {"Z2": {"on": True, "trivial_Z2": even, "symm_modulus": False}},
              "selection": "T",
              "T": {"stage0": {"template": {"Sequivariant": {"ViT2D": {}, "SpinTTNHead": {}, "irrep": [0, 0],
                                                             "phase_mode": "log", "coset": [2, 2]}}}}}
    hydra = Hydra(copy.deepcopy(config), lattice_size=(L, L), hilbert=nk.hilbert.Spin(0.5, L * L))
    x = spins(L)
    p = hydra.model.init(jax.random.key(0), x.reshape(-1, L * L))
    f = lambda y: hydra.model.apply(p, y.reshape(y.shape[0], L * L))
    ref = f(x)
    assert np.all(np.isfinite(np.asarray(ref)))
    for y in (jnp.roll(x, 1, axis=1), jnp.roll(x, 2, axis=2)):
        assert same_log(f(y), ref, tol=1e-8)
    d = np.asarray(f(-x) - ref)
    np.testing.assert_allclose(np.exp(1j * d.imag), 1 if even else -1, atol=1e-8)
    assert np.abs(d.real).max() < 1e-8


@pytest.mark.parametrize("real", [False, True])
def test_float32_close_to_float64(real):
    x = jnp.asarray(np.random.default_rng(7).standard_normal((4, 3, 3, 6)))
    head = SpinTTNHead(token_size=(2, 2), init_scale=0.5, real=real)
    p = head.init(jax.random.key(2), x)
    ref = head.apply(p, x)
    precision.set_network_dtype("float32")
    try:
        low = head.apply(jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32))
    finally:
        precision.set_network_dtype("float64")
    assert low.dtype == jnp.complex64
    assert same_log(low, ref, tol=1e-4)


def test_guards():
    head = SpinTTNHead(token_size=(2, 2))
    with pytest.raises(ValueError, match="feature grid"):
        head.init(jax.random.key(0), jnp.ones((2, 12, 5)))
    with pytest.raises(ValueError, match="row node"):  # 10x10 lattice, token_dim 16: d2 * d1^5 = 8 * 16^5 > 2^20
        SpinTTNHead(token_size=(2, 2), token_dim=16).init(jax.random.key(0), jnp.ones((1, 5, 5, 4)))
