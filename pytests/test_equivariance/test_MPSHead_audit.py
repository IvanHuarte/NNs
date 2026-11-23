"""
MPSHead audit: brute-force contraction, D = 1 limit, long paths in float64/float32,
gradients against finite differences of the definition, and the saddle at A = 1.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_equivariance/test_MPSHead_audit.py -q
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from test_MPSHead import MPSHead, features, same_log  # noqa: E402

from Hydra.NN import precision  # noqa: E402


def site_matrices(params, x, D, n):
    """A^j_r = 1 + LayerNorm(x_r) T^j + b^j in numpy, shape (B, H, W, n, D, D) (flax LayerNorm, fast variance)."""
    p = jax.tree.map(lambda a: np.asarray(a, np.float64), params["params"])
    x = np.asarray(x, np.float64)
    mu = x.mean(-1, keepdims=True)
    var = np.maximum((x ** 2).mean(-1, keepdims=True) - mu ** 2, 0.0)
    y = (x - mu) / np.sqrt(var + 1e-6) * p["LayerNorm_0"]["scale"] + p["LayerNorm_0"]["bias"]
    k, b = p["CDense_0"]["ckernel"], p["CDense_0"]["cbias"]
    m = y @ (k[..., 0] + 1j * k[..., 1]) + (b[:, 0] + 1j * b[:, 1])
    return m.reshape(*x.shape[:3], n, D, D) + np.eye(D)


def reference_log_psi(params, x, D, n):
    """log sum_j sum_c Tr prod_n A^j_{p_n + (0, c)}: explicit products along the row-major torus path
    shifted by c, one (j, c) at a time, rescaled after every step (log domain)."""
    A = site_matrices(params, x, D, n)
    B, H, W = A.shape[:3]
    out = np.empty(B, complex)
    for bi in range(B):
        terms = []
        for j in range(n):
            for c in range(W):
                M, logs = np.eye(D, dtype=complex), 0.0
                for i in range(H):
                    for col in range(W):
                        M = M @ A[bi, i, (col + c) % W, j]
                        s = np.abs(M).max()
                        M, logs = M / s, logs + np.log(s)
                terms.append(logs + np.log(np.trace(M)))
        terms = np.asarray(terms)
        top = terms.real.max()
        out[bi] = top + np.log(np.exp(terms - top).sum())
    return out


def head_and_params(grid, D, n, scale, seed=0, B=4, C=5):
    x = features((B, *grid, C), seed=seed)
    head = MPSHead(bond_dim=D, n_mps=n, init_scale=scale)
    p = head.init(jax.random.key(seed), x)
    # non-trivial LayerNorm affine and bias, so they are checked too
    rng = np.random.default_rng(seed + 100)
    p = jax.tree.map(lambda a: a + 0.3 * rng.standard_normal(a.shape) * scale, p)
    return head, p, x


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("D", [1, 2, 4])
@pytest.mark.parametrize("grid", [(2, 2), (2, 3), (3, 3)])
def test_matches_brute_force(grid, D, n):
    head, p, x = head_and_params(grid, D, n, scale=1.0)  # A far from 1: the order of the products matters
    out = head.apply(p, x)
    assert out.shape == (x.shape[0], 1)
    assert same_log(out, reference_log_psi(p, x, D, n), tol=1e-9)


@pytest.mark.parametrize("grid", [(2, 3), (3, 3)])
def test_invariant_under_every_grid_translation(grid):
    head, p, x = head_and_params(grid, 4, 2, scale=1.0, seed=1)
    ref = head.apply(p, x)
    for dy in range(grid[0]):
        for dx in range(grid[1]):
            assert same_log(head.apply(p, jnp.roll(x, (dy, dx), axis=(1, 2))), ref, tol=1e-9)


@pytest.mark.parametrize("n", [1, 2])
def test_bond_one_is_a_product_of_site_factors(n):
    """D = 1: psi = W sum_j prod_r (1 + t^j.x_r + b^j), independent of the path."""
    head, p, x = head_and_params((3, 4), 1, n, scale=1.0, seed=2)
    a = site_matrices(p, x, 1, n)[..., 0, 0]  # (B, H, W, n)
    W = x.shape[2]
    expected = np.log(W) + np.log(np.exp(np.log(a).sum(axis=(1, 2))).sum(-1))
    assert same_log(head.apply(p, x), expected, tol=1e-9)


@pytest.mark.parametrize("scale", [3.0, 30.0])
def test_long_path_is_stable(scale):
    """5x5 grid (10x10 lattice, 2x2 tokens): 25 products of D = 4 matrices with entries up to ~scale.
    Without rescaling |prod| ~ (2 scale)^25 overflows float32 at scale 30; the head stays finite and close."""
    head, p, x = head_and_params((5, 5), 4, 2, scale=scale, seed=3, B=6)
    ref64 = reference_log_psi(p, x, 4, 2)
    assert same_log(head.apply(p, x), ref64, tol=1e-9)

    p32, x32 = jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32)
    precision.set_network_dtype("float32")
    try:
        low = head.apply(p32, x32)
    finally:
        precision.set_network_dtype("float64")
    assert low.dtype == jnp.complex64 and np.all(np.isfinite(np.asarray(low)))
    # same float32-rounded inputs in float64: what is left is the float32 arithmetic
    ref32 = reference_log_psi(jax.tree.map(lambda a: a.astype(jnp.float64), p32), x32.astype(jnp.float64), 4, 2)
    assert same_log(low, ref32, tol=2e-4)


def test_gradient_matches_finite_differences_of_the_definition():
    """The site and row rescalings carry stop_gradient: the reverse-mode gradient must still be the
    gradient of the plain contraction (checked along random directions, real and imaginary parts)."""
    head, p, x = head_and_params((2, 3), 3, 2, scale=1.0, seed=4, B=3)
    g_re = jax.grad(lambda q: jnp.sum(head.apply(q, x).real))(p)
    g_im = jax.grad(lambda q: jnp.sum(head.apply(q, x).imag))(p)
    rng = np.random.default_rng(5)
    for _ in range(3):
        v = jax.tree.map(lambda a: rng.standard_normal(a.shape), p)
        dot = lambda g: sum(float(jnp.vdot(a, b)) for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(v)))
        h = 1e-6
        f = lambda t: reference_log_psi(jax.tree.map(lambda a, b: a + t * b, p, v), x, 3, 2)
        d = f(h) - f(-h)
        d = (d.real + 1j * ((d.imag + np.pi) % (2 * np.pi) - np.pi)).sum() / (2 * h)
        np.testing.assert_allclose([dot(g_re), dot(g_im)], [d.real, d.imag], rtol=1e-6, atol=1e-8)


def trace_split(g, D, n):
    """|grad| in the trace part (Tr T^a) and in the traceless part of every D x D block of the kernel."""
    g = np.asarray(g).reshape(g.shape[0], n, D, D, 2)
    tr = np.einsum("cjkkr->cjr", g)[:, :, None, None, :] * np.eye(D)[None, None, :, :, None] / D
    return np.linalg.norm(tr), np.linalg.norm(g - tr)


def test_saddle_at_identity():
    """At A = 1 + O(eps) the kernel gradient is the trace part (a linear readout of sum_r x_r) plus a
    traceless part of O(eps): exactly zero at eps = 0, then linear in eps."""
    D, n = 4, 2
    x = features((8, 3, 3, 6), seed=6)
    head = MPSHead(bond_dim=D, n_mps=n, init_scale=1.0)
    p = head.init(jax.random.key(6), x)
    ratios = []
    for eps in (0.0, 1e-3, 1e-2):
        q = jax.tree.map(lambda a: a, p)
        q["params"]["CDense_0"]["ckernel"] = eps * p["params"]["CDense_0"]["ckernel"]
        g = jax.grad(lambda q: jnp.sum(head.apply(q, x).real))(q)["params"]["CDense_0"]["ckernel"]
        tr, tl = trace_split(g, D, n)
        ratios.append(tl / tr)
    assert ratios[0] < 1e-12
    np.testing.assert_allclose(ratios[2] / ratios[1], 10.0, rtol=0.05)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_identical_to_frozen_copy(dtype):
    """Behaviour-preserving changes only: same init and the same log psi and gradient as the frozen
    pre-change MPSHead (_mpshead_frozen.py) on random inputs, grids up to 6x6 (SpinMPSHead paths)."""
    import _mpshead_frozen as old
    low = dtype == "float32"
    for grid, D, n in [((1, 3), 3, 1), ((2, 2), 4, 2), ((3, 3), 8, 2), ((2, 5), 2, 2), ((6, 6), 4, 2)]:
        x = features((3, *grid, 7), seed=7)
        new_h, old_h = MPSHead(bond_dim=D, n_mps=n, init_scale=0.7), old.MPSHead(bond_dim=D, n_mps=n, init_scale=0.7)
        p = new_h.init(jax.random.key(8), x)
        assert jax.tree.all(jax.tree.map(lambda a, b: bool((a == b).all()), p, old_h.init(jax.random.key(8), x)))
        if low:
            p, x = jax.tree.map(lambda a: a.astype(jnp.float32), p), x.astype(jnp.float32)
            precision.set_network_dtype("float32")
            old.DTYPE = jnp.float32
        try:
            f = lambda h: (h.apply(p, x), jax.grad(lambda q: jnp.sum(jnp.abs(h.apply(q, x)) ** 2))(p))
            (yn, gn), (yo, go) = f(new_h), f(old_h)
        finally:
            precision.set_network_dtype("float64")
            old.DTYPE = jnp.float64
        tol = 1e-5 if low else 1e-12
        assert same_log(yn, yo, tol=tol)
        for a, b in zip(jax.tree.leaves(gn), jax.tree.leaves(go)):
            np.testing.assert_allclose(a, b, rtol=tol, atol=tol * float(np.abs(b).max()))


@pytest.mark.parametrize("grid, n", [((2, 2), 1), ((3, 3), 2), ((2, 5), 2)])
def test_nilpotent_init_keeps_the_state(grid, n):
    """init_type="nilpotent": T strictly upper triangular, Tr prod A = D for any init_scale, so log psi =
    log(n W D) on every configuration; the upper (nilpotent) directions get no gradient, the lower ones do."""
    D = 4
    x = features((5, *grid, 7), seed=9)
    head = MPSHead(bond_dim=D, n_mps=n, init_scale=1.0, init_type="nilpotent")
    p = head.init(jax.random.key(9), x)
    np.testing.assert_allclose(head.apply(p, x), np.log(n * grid[1] * D), atol=1e-12)
    g = jax.grad(lambda q: jnp.sum(head.apply(q, x).real))(p)["params"]["CDense_0"]["ckernel"]
    g = np.asarray(g).reshape(7, n, D, D, 2)
    assert np.abs(g * np.triu(np.ones((D, D)), 1)[:, :, None]).max() < 1e-12
    assert np.abs(g * np.tril(np.ones((D, D)), -1)[:, :, None]).max() > 1e-2
    with pytest.raises(ValueError, match="init_type"):
        MPSHead(bond_dim=2, init_type="nilpotnet").init(jax.random.key(0), x)


def test_nilpotent_init_through_storage_keeps_a_warm_start():
    """The option reaches MPSHead and SpinMPSHead from the Hydra storage (gen.py's mps dict): with
    Transversal(OutputHead, head) the log psi differs from the OutputHead-only network by a constant."""
    import copy

    import netket as nk
    from Hydra.NN.Hydra import Hydra
    L = 4
    vit = {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1, "n_ffn_layers": 1,
           "final_architecture": None}

    def model(head, kw):
        storage = {"ViT2D": vit, "OutputHead": {"d_model": 8, "only_phase": False}}
        out = {"OutputHead": {}}
        if head:
            storage[head] = kw
            out = {"Transversal": {"OutputHead": {}, head: {}}}
        config = {"print_arch": False, "load_model": {"parameters": False, "sampler": False}, "storage": storage,
                  "symm_wrapper": {"Z2": {"on": True, "trivial_Z2": True, "symm_modulus": False}}, "selection": "T",
                  "T": {"stage0": {"template": {"Sequivariant": {"ViT2D": {}, **out, "irrep": [0, 0],
                                                                 "phase_mode": "angle", "coset": [2, 2]}}}}}
        return Hydra(copy.deepcopy(config), lattice_size=(L, L), hilbert=nk.hilbert.Spin(0.5, L * L)).model

    x = jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], (6, L * L)))
    m0 = model(None, None)
    p0 = m0.init(jax.random.key(0), x[:1])["params"]
    ref = m0.apply({"params": p0}, x)
    for head, kw in [("MPSHead", {"bond_dim": 4, "n_mps": 2, "init_scale": 1.0, "init_type": "nilpotent"}),
                     ("SpinMPSHead", {"token_size": [2, 2], "bond_dim": 4, "n_mps": 2, "init_scale": 1.0,
                                      "init_type": "nilpotent"})]:
        m = model(head, kw)
        p = m.init(jax.random.key(1), x[:1])["params"]
        p["NN"]["Z2Wrap"]["SeqV_0"] = p0["NN"]["Z2Wrap"]["SeqV_0"]
        p["NN"]["Z2Wrap"]["SeqV_1"]["Trans_0"] = p0["NN"]["Z2Wrap"]["SeqV_1"]
        d = np.asarray(m.apply({"params": p}, x) - ref)
        assert np.ptp(d.real) < 1e-10 and np.ptp(d.imag) < 1e-10


def test_spin_flags_share_the_mpshead_contraction():
    """SpinMPSHead(real=True) contracts with MPSHead's own mps_chain (no diverging copy)."""
    import importlib
    spin = importlib.import_module("Hydra.NN.NN.SingleModules.SpinMPSHead")
    mps = importlib.import_module("Hydra.NN.NN.SingleModules.MPSHead")
    assert spin.mps_chain is mps.mps_chain


@pytest.mark.parametrize("flag", ["marshall", "physical"])
def test_spin_flags_refuse_a_backbone_without_spins(flag):
    """marshall/physical read the last sy*sx channels as spins: with plain ViT2D the Hydra build (eager init)
    raises, and a jitted call on non-spin channels gives NaN instead of a silently wrong state."""
    import copy

    import netket as nk
    from Hydra.NN.Hydra import Hydra
    from Hydra.NN.NN.SingleModules.SpinMPSHead import SpinMPSHead
    from Hydra.NN.NN.SingleModules.ViT2D import Tokenize
    L = 4
    vit = {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1, "n_ffn_layers": 1,
           "final_architecture": None}

    def build(backbone):
        storage = {backbone: vit, "SpinMPSHead": {"token_size": [2, 2], "bond_dim": 3, "features": 4,
                                                  "init_scale": 0.3, flag: True}}
        config = {"print_arch": False, "load_model": {"parameters": False, "sampler": False}, "storage": storage,
                  "symm_wrapper": {"Z2": {"on": True, "trivial_Z2": True, "symm_modulus": False}}, "selection": "T",
                  "T": {"stage0": {"template": {"Sequivariant": {backbone: {}, "SpinMPSHead": {}, "irrep": [0, 0],
                                                                 "phase_mode": "angle", "coset": [2, 2]}}}}}
        return Hydra(copy.deepcopy(config), lattice_size=(L, L), hilbert=nk.hilbert.Spin(0.5, L * L))

    with pytest.raises(ValueError, match="ViT2DSpins"):
        build("ViT2D")
    build("ViT2DSpins")  # the intended pairing builds

    head = SpinMPSHead(token_size=(2, 2), bond_dim=3, features=4, init_scale=0.3, **{flag: True})
    feats = features((3, 2, 2, 9), seed=11)
    spins = Tokenize((2, 2), (2, 2), 4)(jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], (3, 4, 4))))
    good = jnp.concatenate([feats[..., :5], spins], -1)
    p = head.init(jax.random.key(0), good)
    assert np.all(np.isfinite(np.asarray(jax.jit(head.apply)(p, good))))
    assert np.all(np.isnan(np.asarray(jax.jit(head.apply)(p, feats))))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_real_chain_exact_zero_term_has_a_finite_gradient(dtype):
    """Real site matrices (SpinMPSHead(real=True)): a term whose trace is exactly 0 left psi finite but gave a
    NaN gradient (log per term + logsumexp: 0 * inf), which diverged E01 (slurm 1509). Now finite, and the real
    path still equals the plain sum of the terms; an exactly zero sum is finite too."""
    import importlib
    chain = importlib.import_module("Hydra.NN.NN.SingleModules.MPSHead").mps_chain
    zero_term = jnp.asarray(np.array([[[[[[1.0, 0.3], [0.2, -1.0]], [[1.0, 0.0], [0.0, 1.0]]]]]]), dtype)
    zero_sum = jnp.asarray(np.array([[[[[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]]]]]), dtype)
    for a in (zero_term, zero_sum):
        g = jax.grad(lambda m: chain(m)[0, 0].real)(a)
        assert np.all(np.isfinite(np.asarray(g))) and np.isfinite(complex(chain(a)[0, 0]))
    assert abs(complex(chain(zero_term)[0, 0]) - np.log(2.0)) < 1e-6
    rng = np.random.default_rng(12)
    m = rng.standard_normal((3, 2, 3, 2, 3, 3)) * 0.4 + np.eye(3)
    ref = np.zeros(3, complex)
    for b in range(3):
        for j in range(2):
            for c in range(3):
                M = np.eye(3)
                for i in range(2):
                    for col in range(3):
                        M = M @ m[b, i, (col + c) % 3, j]
                ref[b] += np.trace(M)
    assert same_log(chain(jnp.asarray(m, dtype)), np.log(ref.astype(complex)), tol=1e-9 if dtype == jnp.float64 else 1e-5)
