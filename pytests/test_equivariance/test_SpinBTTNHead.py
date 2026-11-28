"""
SpinBTTNHead: SpinTTNHead with a binary tree (bisection of the token grid) above the token nodes.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_equivariance/test_SpinBTTNHead.py -q
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

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from test_SpinTTNHead import leaves_of, same_log, spins  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN import precision  # noqa: E402
from Hydra.NN.NN.FluxModules.Sequivariant import Sequivariant  # noqa: E402
from Hydra.NN.NN.SingleModules.SpinBTTNHead import (  # noqa: E402
    SpinBTTNHead,
    bisection_tree,
    btree_amplitude,
    btree_log_psi,
    tree_dims,
)
from Hydra.NN.NN.SingleModules.SpinTTNHead import SpinTTNHead  # noqa: E402
from Hydra.NN.NN.SingleModules.ViT2D import ViT2D  # noqa: E402

INITS = ["identity", "isometry"]


def rand(shape, cplx, rng):
    return rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)


def random_tree(H, W, d1, bond, cplx, rng):
    tree = bisection_tree(H, W)
    chi = tree_dims(tree, H * W, d1, bond)
    return tree, [rand((chi[H * W + k], chi[l] * chi[r]), cplx, rng) for k, (l, r, _) in enumerate(tree)]


def naive_tree(u, nodes, tree, xp=np):
    """Amplitude of the tree on a token grid u (H, W, d1): one einsum over every bond, no rescaling."""
    H, W, d1 = u.shape
    P = H * W
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    dims = [d1] * P + [T.shape[0] for T in nodes]
    specs, ops = [letters[p] for p in range(P)], [u[p // W, p % W] for p in range(P)]
    for k, ((l, r, _), T) in enumerate(zip(tree, nodes)):
        root = k == len(tree) - 1
        T = T.reshape(T.shape[0], dims[l], dims[r])
        specs.append(("" if root else letters[P + k]) + letters[l] + letters[r])
        ops.append(T[0] if root else T)
    return xp.einsum(",".join(specs) + "->", *ops, optimize="greedy")


def naive_log_psi(leaves, w1, nodes, tree, xp=np):
    """Token node on the leaves (one einsum per token), then the sum of the tree over the H*W cyclic shifts."""
    B, H, W, K, d0 = leaves.shape
    w = w1.reshape((w1.shape[0],) + (d0,) * K)
    spec = "z" + "abcdefghij"[:K] + "," + ",".join("abcdefghij"[:K]) + "->z"
    u = [xp.stack([xp.stack([xp.einsum(spec, w, *[leaves[b, i, j, k] for k in range(K)]) for j in range(W)])
                   for i in range(H)]) for b in range(B)]
    psi = [sum(naive_tree(xp.roll(u[b], (-a, -c), axis=(0, 1)), nodes, tree, xp) for a in range(H) for c in range(W))
           for b in range(B)]
    return xp.log(xp.stack(psi).astype(complex))[:, None]


def node_names(tree):
    return [f"w_r{i0}c{j0}_{h}x{w}" for _, _, (i0, j0, h, w) in tree]


def nodes_of(p, tree, real):
    """w_token and the tree nodes (in the order of tree) of the module parameters."""
    q = p["params"]
    return [q[n] if real else q[n][..., 0] + 1j * q[n][..., 1] for n in ["w_token"] + node_names(tree)]


def log_spread(y):
    """Std of log|psi| and circular std of the phase over the batch."""
    y = np.asarray(y).ravel()
    return y.real.std(), np.sqrt(-2 * np.log(np.abs(np.exp(1j * y.imag).mean())))


# ---------------------------------------------------------------- tree layout


def test_bisection_tree_of_the_3x3_grid():
    """6x6 lattice, tokens 2x2: the layout drawn in the module docstring (children first, root last)."""
    assert [blk for _, _, blk in bisection_tree(3, 3)] == [
        (1, 0, 2, 1), (0, 0, 3, 1), (0, 1, 1, 2), (1, 1, 2, 1), (1, 2, 2, 1), (1, 1, 2, 2), (0, 1, 3, 2), (0, 0, 3, 3)]


@pytest.mark.parametrize("H, W", [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (3, 7)])
def test_bisection_tree_splits(H, W):
    """N_t - 1 nodes, children first; a block splits along its longer side (columns on a tie) into its floor
    (top/left) and ceil halves; every token is a leaf once and the root is the whole grid."""
    P, tree = H * W, bisection_tree(H, W)
    assert len(tree) == P - 1
    block, leaves = {i * W + j: (i, j, 1, 1) for i in range(H) for j in range(W)}, []
    for k, (l, r, (i0, j0, h, w)) in enumerate(tree):
        assert l in block and r in block
        if w >= h:
            assert block[l] == (i0, j0, h, w // 2) and block[r] == (i0, j0 + w // 2, h, w - w // 2)
        else:
            assert block[l] == (i0, j0, h // 2, w) and block[r] == (i0 + h // 2, j0, h - h // 2, w)
        leaves += [c for c in (l, r) if c < P]
        block[P + k] = (i0, j0, h, w)
    assert block[2 * P - 2] == (0, 0, H, W) and sorted(leaves) == list(range(P))


def test_bond_dims():
    """chi = d1 on the tokens, min(bond, chi_left chi_right) inside, 1 at the root (3x3 grid)."""
    tree = bisection_tree(3, 3)
    assert tree_dims(tree, 9, 8, 32)[9:] == [32] * 7 + [1]
    assert tree_dims(tree, 9, 8, 8)[9:] == [8] * 7 + [1]
    assert tree_dims(tree, 9, 2, 32)[9:] == [4, 8, 4, 4, 4, 16, 32, 1]


# ---------------------------------------------------------------- exact contraction


@pytest.mark.parametrize("cplx", [True, False])
@pytest.mark.parametrize("H, W", [(2, 2), (3, 3), (5, 5), (2, 3), (1, 3)])
def test_tree_amplitude_matches_brute_force(H, W, cplx):
    """One tree, no shift sum: 4x4, 6x6, 10x10 lattices and non-square grids. d1 = 3, bond 5: the levels mix
    node shapes (several einsums per level)."""
    rng = np.random.default_rng(0)
    tree, nodes = random_tree(H, W, 3, 5, cplx, rng)
    u = rand((2, H, W, 3), cplx, rng)
    t, log_t = btree_amplitude(jnp.asarray(u.reshape(2, H * W, 3).transpose(1, 0, 2)), list(map(jnp.asarray, nodes)),
                               tree)
    ref = np.log(np.array([naive_tree(u[b], nodes, tree) for b in range(2)], complex))
    assert same_log(log_t + jnp.log(t.astype(jnp.complex128)), ref, tol=1e-10)


@pytest.mark.parametrize("cplx", [True, False])
@pytest.mark.parametrize("H, W, bond", [(2, 2, 3), (3, 3, 5), (2, 3, 4), (5, 5, 3)])
def test_log_psi_matches_brute_force(H, W, bond, cplx):
    """Leaves -> token node -> tree, summed over the H*W grid shifts, against explicit einsums."""
    rng = np.random.default_rng(1)
    K, d0, d1 = 4, 2, 3
    leaves, w1 = rand((2, H, W, K, d0), cplx, rng), rand((d1, d0**K), cplx, rng)
    tree, nodes = random_tree(H, W, d1, bond, cplx, rng)
    out = btree_log_psi(jnp.asarray(leaves), jnp.asarray(w1), list(map(jnp.asarray, nodes)), tree)
    assert out.shape == (2, 1)
    assert same_log(out, naive_log_psi(leaves, w1, nodes, tree), tol=1e-10)


def test_gradient_matches_plain_contraction():
    """The rescaling (stop_gradient) and the recomputed outer products leave the gradient of the plain einsums."""
    rng = np.random.default_rng(2)
    H, W, K, d0, d1 = 3, 3, 4, 2, 3
    leaves, w1 = rand((2, H, W, K, d0), True, rng), rand((d1, d0**K), True, rng)
    tree, nodes = random_tree(H, W, d1, 4, True, rng)
    args = (jnp.asarray(leaves), jnp.asarray(w1), list(map(jnp.asarray, nodes)))

    def loss(f):
        return lambda lv, w, ns: jnp.sum(jnp.real(f(lv, w, ns, tree)) + 0.3 * jnp.imag(f(lv, w, ns, tree)))

    g = jax.grad(loss(btree_log_psi), argnums=(0, 1, 2))(*args)
    ref = jax.grad(loss(lambda lv, w, ns, tr: naive_log_psi(lv, w, ns, tr, jnp)), argnums=(0, 1, 2))(*args)
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("init_type", INITS)
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("g", [2, 3])
def test_module_is_the_tree_of_its_leaves(g, real, init_type):
    x = jnp.asarray(np.random.default_rng(1).standard_normal((3, g, g, 7)))
    head = SpinBTTNHead(token_size=(2, 2), leaf_dim=3, token_dim=4, bond_dim=6, features=5, init_scale=0.7, real=real,
                        init_type=init_type)
    p = head.init(jax.random.key(0), x)
    out = head.apply(p, x)
    assert out.shape == (3, 1) and out.dtype == jnp.complex128
    tree = bisection_tree(g, g)
    w1, *nodes = map(np.asarray, nodes_of(p, tree, real))
    assert same_log(out, naive_log_psi(np.asarray(leaves_of(head, p, x)), w1, nodes, tree), tol=1e-10)
    if real:
        assert np.abs(np.sin(np.asarray(out.imag))).max() < 1e-12  # real psi: phase 0 or pi


def test_leaf_and_token_stage_is_the_spin_ttn_one():
    """Same seed: the leaf map and the token node of SpinTTNHead (names, shapes and values)."""
    x = jnp.asarray(np.random.default_rng(3).standard_normal((2, 3, 3, 6)))
    p = SpinBTTNHead(token_size=(2, 2)).init(jax.random.key(4), x)["params"]
    q = SpinTTNHead(token_size=(2, 2)).init(jax.random.key(4), x)["params"]
    for k in ("Dense_0", "LayerNorm_0", "CDense_0", "w_token"):
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(p[k]), jax.tree.leaves(q[k])))


def test_tree_node_init():
    """identity: T[0, 0, 0] = 1 + init_scale noise (exactly e_000 at init_scale 0); isometry: T T^dagger = 1."""
    x = jnp.ones((1, 3, 3, 4))
    tree = bisection_tree(3, 3)
    for init_scale in (0.0, 0.05):
        p = SpinBTTNHead(token_size=(2, 2), init_scale=init_scale).init(jax.random.key(0), x)
        for T in nodes_of(p, tree, False)[1:]:
            noise = np.asarray(T).copy()
            noise[0, 0] -= 1
            if init_scale:
                assert 0.7 < noise.real.std() / init_scale < 1.3 and 0.7 < noise.imag.std() / init_scale < 1.3
            else:
                assert not noise.any()
    p = SpinBTTNHead(token_size=(2, 2), init_type="isometry").init(jax.random.key(0), x)
    for T in nodes_of(p, tree, False)[1:]:
        T = np.asarray(T)
        np.testing.assert_allclose(T @ T.conj().T, np.eye(T.shape[0]), atol=1e-12)


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("init_type", INITS)
def test_tree_scale_is_an_exact_reparametrization(init_type, real):
    """tree_scale s: tree nodes stored as T / s (same T, same psi) and their gradients s times larger."""
    s = 10.0
    x = jnp.asarray(np.random.default_rng(4).standard_normal((3, 3, 3, 6)))
    tree = bisection_tree(3, 3)
    names = node_names(tree)
    kw = dict(token_size=(2, 2), bond_dim=6, init_scale=0.3, init_type=init_type, real=real)
    h1, hs = SpinBTTNHead(**kw), SpinBTTNHead(**kw, tree_scale=s)
    p1, ps = h1.init(jax.random.key(5), x), hs.init(jax.random.key(5), x)
    for n in names:
        np.testing.assert_allclose(np.asarray(ps["params"][n]) * s, np.asarray(p1["params"][n]), rtol=1e-12, atol=1e-14)
    for k in set(p1["params"]) - set(names):  # leaves and token node untouched
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(p1["params"][k]), jax.tree.leaves(ps["params"][k])))
    assert same_log(hs.apply(ps, x), h1.apply(p1, x), tol=1e-10)
    for part in (jnp.real, jnp.imag):
        g1 = jax.grad(lambda p: part(h1.apply(p, x)).sum())(p1)["params"]
        gs = jax.grad(lambda p: part(hs.apply(p, x)).sum())(ps)["params"]
        for n in names:
            np.testing.assert_allclose(np.asarray(gs[n]), s * np.asarray(g1[n]), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(np.asarray(gs["w_token"]), np.asarray(g1["w_token"]), rtol=1e-8, atol=1e-12)
    with pytest.raises(ValueError, match="tree_scale"):
        SpinBTTNHead(token_size=(2, 2), tree_scale=0.0).init(jax.random.key(0), x)


# ---------------------------------------------------------------- symmetry


@pytest.mark.parametrize("init_type", INITS)
@pytest.mark.parametrize("H, W", [(2, 2), (3, 3), (2, 3), (5, 5)])
def test_invariant_under_cyclic_shifts_of_the_feature_grid(H, W, init_type):
    x = jnp.asarray(np.random.default_rng(2).standard_normal((3, H, W, 6)))
    head = SpinBTTNHead(token_size=(2, 2), bond_dim=6, init_scale=0.7, init_type=init_type)
    p = head.init(jax.random.key(1), x)
    ref = head.apply(p, x)
    for shift in ((1, 0), (0, 1), (1, 2), (H - 1, W - 1)):
        assert same_log(head.apply(p, jnp.roll(x, shift, axis=(1, 2))), ref, tol=1e-10)
    if H * W >= 9:  # but not a pooled readout: a generic permutation of the tokens changes it
        perm = np.random.default_rng(3).permutation(H * W)
        y = x.reshape(3, H * W, 6)[:, perm].reshape(x.shape)
        assert not same_log(head.apply(p, y), ref, tol=1e-6)


@pytest.mark.parametrize("L, irrep", [(4, (0, 0)), (4, (1, 2)), (6, (0, 0)), (6, (3, 1)), (10, (0, 0)), (10, (2, 3))])
def test_sequivariant_coset_characters(L, irrep):
    """ViT2D tokens 2x2 + head under Sequivariant, coset (2, 2): shifts by one site and by one token give
    exp(2 pi i k.t / L), finite log psi and gradient; 6x6 and 10x10 have odd token grids (3x3, 5x5)."""
    vit = ViT2D(token_size=(2, 2), embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    m = Sequivariant(SeqV=(vit, SpinBTTNHead(token_size=(2, 2), bond_dim=6, init_scale=0.5)), irrep=irrep,
                     coset=(2, 2), phase_mode="log")
    x = spins(L)
    p = m.init(jax.random.key(0), x)
    f = jax.jit(m.apply)
    out = f(p, x)
    assert out.shape == (x.shape[0], 1) and np.all(np.isfinite(np.asarray(out)))
    for axis, k in ((1, irrep[0]), (2, irrep[1])):
        for t in (1, 2):
            d = np.asarray(f(p, jnp.roll(x, t, axis=axis)) - out)
            assert np.abs(d.real).max() < 1e-9
            np.testing.assert_allclose(np.exp(1j * d.imag), np.exp(2j * np.pi * k * t / L), atol=1e-9)
    g = jax.jit(jax.grad(lambda q: jnp.sum(m.apply(q, x).real)))(p)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(g))


@pytest.mark.parametrize("L, even", [(4, True), (4, False), (6, True), (6, False)])
def test_built_by_hydra_with_z2(L, even):
    """Registry + storage (token_size as a list), coset (2,2), Z2 even/odd: translations and the spin flip."""
    from Hydra.NN.Hydra import Hydra
    storage = {"ViT2D": {"token_size": [2, 2], "embedding_d": 8, "n_heads": 2, "n_blocks": 1, "n_ffn_layers": 1,
                         "final_architecture": None},
               "SpinBTTNHead": {"token_size": [2, 2], "leaf_dim": 4, "token_dim": 8, "bond_dim": 12, "init_scale": 0.5,
                                "init_type": "isometry"}}
    config = {"print_arch": False, "load_model": {"parameters": False, "sampler": False}, "storage": storage,
              "symm_wrapper": {"Z2": {"on": True, "trivial_Z2": even, "symm_modulus": False}},
              "selection": "T",
              "T": {"stage0": {"template": {"Sequivariant": {"ViT2D": {}, "SpinBTTNHead": {}, "irrep": [0, 0],
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


def test_label_and_registry():
    from Hydra.label_utils import HEAD_ABBR, experiment_name
    from Hydra.NN import REGISTRY
    assert REGISTRY["SpinBTTNHead"] is SpinBTTNHead
    assert experiment_name("ViT2D", "SpinBTTNHead", (2, 2), 1, "_0") == "ViT2DSBTTN221_0"
    assert len(set(HEAD_ABBR.values())) == len(HEAD_ABBR)


# ---------------------------------------------------------------- numerics, init, dtypes


@pytest.mark.parametrize("low", [False, True])
def test_log_scale_stability(low):
    """Multilinear: leaf n times e^a_n -> log psi + sum a_n; w1 times l -> l^(H W); node n times l_n -> + log l_n.
    10x10 lattice (5x5 grid), no overflow in complex64 either."""
    H = W = 5
    rng = np.random.default_rng(3)
    leaves, w1 = rand((2, H, W, 4, 2), True, rng), rand((3, 16), True, rng)
    tree, nodes = random_tree(H, W, 3, 4, True, rng)
    base = btree_log_psi(jnp.asarray(leaves), jnp.asarray(w1), list(map(jnp.asarray, nodes)), tree)
    a = rng.uniform(-1, 1, leaves.shape[:4]) * (30 if low else 300)
    lam = 10.0 ** (rng.uniform(-1, 1, len(nodes)) * (15 if low else 100))
    cases = [((leaves * np.exp(a)[..., None], w1, nodes), a.sum((1, 2, 3))[:, None]),
             ((leaves, w1 * 1e-6, [T * s for T, s in zip(nodes, lam)]), np.log(1e-6) * H * W + np.log(lam).sum())]
    dt = jnp.complex64 if low else jnp.complex128
    for (lv, w, ns), shift in cases:
        out = btree_log_psi(jnp.asarray(lv, dt), jnp.asarray(w, dt), [jnp.asarray(T, dt) for T in ns], tree)
        assert np.all(np.isfinite(np.asarray(out)))
        assert same_log(np.asarray(out, np.complex128) - shift, base, tol=2e-3 if low else 1e-8)


@pytest.mark.parametrize("init_type", INITS)
def test_flat_log_psi_at_zero_init_scale(init_type):
    x = jnp.asarray(np.random.default_rng(5).standard_normal((16, 3, 3, 8)))
    head = SpinBTTNHead(token_size=(2, 2), init_scale=0.0, init_type=init_type)
    out = np.asarray(head.apply(head.init(jax.random.key(0), x), x))
    assert np.ptp(out.real) < 1e-12 and np.ptp(out.imag) < 1e-12


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("init_type", INITS)
def test_init_flatness(init_type, dtype):
    """At init log psi varies over the configurations through the leaves only, linearly in init_scale."""
    x = jnp.asarray(np.random.default_rng(6).standard_normal((64, 3, 3, 8)))

    def spread(scale):
        head = SpinBTTNHead(token_size=(2, 2), init_scale=scale, init_type=init_type)
        p = head.init(jax.random.key(0), x)
        if dtype == "float64":
            return np.array(log_spread(head.apply(p, x)))
        precision.set_network_dtype("float32")
        try:
            return np.array(log_spread(head.apply(jax.tree.map(lambda a: a.astype(jnp.float32), p),
                                                  x.astype(jnp.float32))))
        finally:
            precision.set_network_dtype("float64")

    s1, s2 = spread(0.005), spread(0.01)
    assert np.all(s2 < 0.5) and np.all((1.6 < s2 / s1) & (s2 / s1 < 2.4)), (s1, s2)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("init_type", INITS)
@pytest.mark.parametrize("real", [False, True])
def test_finite_gradients_at_init_and_at_random_parameters(real, init_type, dtype):
    low = jnp.float32 if dtype == "float32" else jnp.float64
    x = jnp.asarray(np.random.default_rng(2).standard_normal((4, 3, 3, 6)), low)
    head = SpinBTTNHead(token_size=(2, 2), real=real, init_type=init_type)  # init_scale only enters the init
    inits = [head.clone(init_scale=s).init(jax.random.key(0), x) for s in (0.0, 0.05)]
    noisy = [jax.tree.map(lambda v: v + 0.5 * jax.random.normal(jax.random.key(9), v.shape, v.dtype), p) for p in inits]
    kernel = ("Dense_1", "kernel") if real else ("CDense_0", "ckernel")
    if dtype == "float32":
        precision.set_network_dtype("float32")
    try:
        jac = jax.jit(jax.jacrev(lambda t: jnp.stack([jnp.sum(head.apply(t, x).real), jnp.sum(head.apply(t, x).imag)])))
        for k, q in enumerate(inits + noisy):
            g = jac(jax.tree.map(lambda a: a.astype(low), q))
            assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(g))
            if k < len(inits):  # flat start is not a dead point: every tree node and the leaf kernel get a gradient
                gp = g["params"]
                assert all(np.abs(np.asarray(gp[n][0])).max() > 0 for n in node_names(bisection_tree(3, 3)))
                assert np.abs(np.asarray(gp[kernel[0]][kernel[1]][0])).max() > 0
    finally:
        precision.set_network_dtype("float64")


def test_jit_vmap_and_lax_map_over_chunks():
    x = jnp.asarray(np.random.default_rng(6).standard_normal((12, 3, 3, 6)))
    head = SpinBTTNHead(token_size=(2, 2), init_scale=0.5)
    p = head.init(jax.random.key(0), x)
    ref = head.apply(p, x)
    f = lambda c: head.apply(p, c)
    chunks = x.reshape(3, 4, 3, 3, 6)
    for out in (jax.jit(head.apply)(p, x), jax.vmap(f)(chunks).reshape(12, 1), jax.lax.map(f, chunks).reshape(12, 1)):
        assert same_log(out, ref, tol=1e-12)


def test_runs_under_shard_map():
    """netket evaluates the model inside shard_map."""
    from jax.sharding import Mesh, PartitionSpec as P
    x = jnp.asarray(np.random.default_rng(8).standard_normal((2, 3, 3, 5)))
    head = SpinBTTNHead(token_size=(2, 2), init_scale=0.5)
    p = head.init(jax.random.key(0), x)
    mesh = Mesh(np.array(jax.devices()[:1]), ("S",))
    f = jax.shard_map(head.apply, mesh=mesh, in_specs=(P(), P("S")), out_specs=P("S"))
    assert same_log(f(p, x), head.apply(p, x), tol=1e-12)


@pytest.mark.parametrize("init_type", INITS)
@pytest.mark.parametrize("real", [False, True])
def test_float32_close_to_float64(real, init_type):
    x = jnp.asarray(np.random.default_rng(7).standard_normal((4, 3, 3, 6)))
    head = SpinBTTNHead(token_size=(2, 2), bond_dim=16, init_scale=0.5, real=real, init_type=init_type)
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
    head = SpinBTTNHead(token_size=(2, 2))
    with pytest.raises(ValueError, match="feature grid"):
        head.init(jax.random.key(0), jnp.ones((2, 12, 5)))
    with pytest.raises(ValueError, match="at least 2 tokens"):
        head.init(jax.random.key(0), jnp.ones((2, 1, 1, 5)))
    with pytest.raises(ValueError, match="init_type"):
        SpinBTTNHead(token_size=(2, 2), init_type="random").init(jax.random.key(0), jnp.ones((1, 2, 2, 4)))
    with pytest.raises(ValueError, match=r"\[5x2\] node"):  # 10x10, bond 128: (128, 128 x 128) > 2^20 entries
        SpinBTTNHead(token_size=(2, 2), bond_dim=128).init(jax.random.key(0), jnp.ones((1, 5, 5, 4)))
