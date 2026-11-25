"""
SpinTTNHead: tree-tensor-network readout with one leaf per spin (backflow TTN).

Design and rationale: Simulations_claude/explorer/papers.md section 5 (MERA heads). Fan et al.,
arXiv:2603.14425, use the network as a global disentangler and a tensor network with
feature-dependent tensors for the residual correlations, and name tree tensor networks as an
upgrade of their MPS back end. A MERA adds disentanglers, whose loops make the full amplitude
<s|MERA> expensive (perfect sampling, Ferris and Vidal PRB 85, 165146/165147, needs the spins as
the leaves of a unitary network, which a backflow head is not). Here the ViT is the disentangler
and the tensor network is a tree (Tagliacozzo, Evenbly, Vidal, PRB 80, 235127; a hierarchical
product pooling, Cohen, Sharir, Shashua 2016).

    leaves   v_n = e0 + K . LN(SiLU(W_ab . x_t + c_ab))           one per spin, dimension d0
    token    u_t = w1 (v_t1 x ... v_t(sy*sx))                     w1: d1 x d0^(sy*sx), shared
    rows     r_i^c = w2 (u_(i,c) x u_(i,c+1) x ... x u_(i,c+W-1))   w2: d2 x d1^W, column shift c
    top      psi = sum_{c<W} sum_{s<H} w3 (r_s^c x r_(s+1)^c x ... x r_(s+H-1)^c)    w3: d2^H

Translations: the leaves and token nodes are the same for every token (equivariant), and psi
sums the tree over the H*W cyclic shifts of the token grid, so it is exactly invariant under the
token-grid translations, like OutputHead/SpinMPSHead. The coset of Sequivariant adds the
intra-token translations; Z2 and C4v act outside. The head reads no spins (no physical index or
sign rule), so any feature grid works (ViT2D, CvT3).

Numerics: every node output is rescaled by its max (stop_gradient, exact) and the scales are
summed in log space. The H*W shift terms are added as numbers and a single log is taken, with the
regulator of mps_chain (a per-term log + logsumexp has a NaN gradient at a zero term).

Init: K small (init_scale) and bias e0, so every leaf is ~e0 and log psi (modulus and phase) is
flat over configurations; w1, w2, w3 are random isometries (QR), i.e. generic environments. To
first order the change of log psi is a slot-dependent linear readout of the leaves, and to second
order it has tree-dependent pair terms (same token, same row, other) - not the pooled saddle of an
MPS at the identity (papers.md section 0).

Input: (B, H, W, C) feature grid of a backbone with final_architecture=None and token/stride
token_size. Output: (B, 1) complex log psi (real tensors: real psi, phase 0 or pi).

Cost per sample (complex MACs): N h d0 (leaves) + H W d1 d0^(sy sx) (tokens)
+ W H d2 d1^W (rows, all column shifts) + W H d2^H (top). 6x6 with tokens 2x2, d0 4, d1 d2 8:
~7.5e4, about half of SpinMPSHead D8 n2.

Tensor-network drawing, 6x6 lattice: tokens 2x2 -> token grid H = W = 3, K = sy*sx = 4,
d0 = leaf_dim = 4, d1 = token_dim = 8, d2 = row_dim = 8, h = features = 16. Boxes are tensors, lines
are indices (labels = dimension); the first drawing is the tree of the grid shift (c, s) = (0, 0).

  ┌────────────────────────────────────────────────────────────────────────────────┐
  │ x = ViT2D(spins): (B, H=3, W=3, C=60)  - one C-leg per token t_ij              │   <---- 1
  │  t00      t01      t02      t10      t11      t12      t20      t21      t22   │
  └───┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────┘
      │        │        │        │        │        │        │        │        │ C
    ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐
    │FFN │   │FFN │   │FFN │   │FFN │   │FFN │   │FFN │   │FFN │   │FFN │   │FFN │     <---- 2
    └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘
     ││││     ││││     ││││     ││││     ││││     ││││     ││││     ││││     ││││ K*h
    ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐
    │ K  │   │ K  │   │ K  │   │ K  │   │ K  │   │ K  │   │ K  │   │ K  │   │ K  │     <---- 2
    └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘   └┬┬┬┬┘
     vvvv     vvvv     vvvv     vvvv     vvvv     vvvv     vvvv     vvvv     vvvv d0   <---- 3
    ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐   ┌┴┴┴┴┐
    │ w1 │   │ w1 │   │ w1 │   │ w1 │   │ w1 │   │ w1 │   │ w1 │   │ w1 │   │ w1 │     <---- 4
    └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
      u        u        u        u        u        u        u        u        u d1     <---- 5
    ┌─┴────────┴────────┴─┐    ┌─┴────────┴────────┴─┐    ┌─┴────────┴────────┴─┐
    │      w2 (row 0)     │    │      w2 (row 1)     │    │      w2 (row 2)     │      <---- 6
    └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘
               r                          r                          r d2              <---- 7
           ┌───┴──────────────────────────┴──────────────────────────┴───┐             <---- 8
           │          w3  ->  t[c=0, s=0]  (scalar, no out leg)          │             <---- 9
           └─────────────────────────────────────────────────────────────┘

  Other trees: the same boxes re-wired by the grid shift (c, s); 9 trees in all.
    column shift c:  w2 of row i reads U[:, c, i, :] = u[:, i, (j + c) % 3]            <---- 5
        c = 1:   u_i1      u_i2      u_i0          (c = 2: u_i2, u_i0, u_i1)
                   └─────────┼─────────┘
                    w2 (row i) -> r[:, c, i], rescaled                                 <---- 6
    row shift s:     w3 reads R[:, c, s, :] = r[:, c, (i + s) % 3]                     <---- 8
        s = 1:   r_1       r_2       r_0           (s = 2: r_2, r_0, r_1)
                   └─────────┼─────────┘
                        w3 -> t[c, s]                                                  <---- 9
    psi_head = exp(log_c) * sum_{c<3, s<3} exp(log_t[c]) t[c, s]   (one log)           <---- 10
    Z2( Sequivariant( ViT2D -> SpinTTNHead, coset (2,2) ) )                            <---- 11

Steps (shapes for a batch B; names as in the code):
  1  Backbone: x = ViT2D(spins) with final_architecture=None, (B, H=3, W=3, C=60), one C-leg per
     token t_ij. Sequivariant reads channel 0 of x for the momentum factor z.
  2  Leaf map (FFN + K): Dense_0 (C -> K*h = 4*16) . SiLU . reshape (B, 3, 3, K=4, h) . LayerNorm_0,
     then CDense_0 (h -> d0 = leaf_dim = 4, bias e0; Dense_1 if real): one leaf v per spin,
     leaves (B, 3, 3, 4, 4); slot k = a*sx + b is the spin (2i + a, 2j + b).
  3  Rescale the leaves: v <- v / max|v| (stop_gradient, exact); log_c = sum of the 36 log scales
     (common to every tree: each tree uses each leaf once).
  4  Token node: u = node_apply(w1 = w_token: d1 x d0^4, [v_0, v_1, v_2, v_3]), (B, 3, 3, d1 = 8),
     the same w_token for every token; rescaled, log_c += the 9 token log scales.
  5  Column shift: U = stack_c roll(u, -c, axis=2), U[:, c, i, j] = u[:, i, j + c], (B, 3, 3, 3, d1).
  6  Row node: r = node_apply(w2 = w_row: d2 x d1^3, [U[..., 0, :], U[..., 1, :], U[..., 2, :]]),
     (B, c = 3, rows = 3, d2 = 8): one contraction per row and column shift.
  7  Rescale the rows: log_t[c] = sum over the 3 rows of their log scales, (B, 3) (each row shift
     uses every row once, so it depends on c only).
  8  Row shift: R = stack_s roll(r, -s, axis=2), R[:, c, s, i] = r[:, c, i + s], (B, 3, 3, 3, d2).
  9  Top node: t = node_apply(w3 = w_top: 1 x d2^3, [R[..., 0, :], R[..., 1, :], R[..., 2, :]]),
     (B, 3, 3): the scalar tree amplitude of each grid shift (c, s).
 10  Regulated log-sum, no per-term log: m = max_c log_t, t <- exp(log_t - m) t,
     A = sum_{c,s} t, S = sum_{c,s} |t|, log psi_head = log_c + m + log(A + 1j REAL_NODE_EPS S),
     (B, 1) complex. The H*W shift sum makes it invariant under token-grid translations.
 11  Wrappers: Sequivariant multiplies by the momentum factor of z (angle/log) and sums the 4
     coset images (intra-token rolls) with exp(-i k.t); Z2 sums s and -s (even/odd). C4v, if on,
     is outermost.

4x4 lattice (token grid H = W = 2): the same drawing with 4 tokens, 2 rows of 2 (w_row: d2 x d1^2),
w_top: 1 x d2^2, column and row shifts c, s < 2, so 4 trees instead of 9.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ...toolbox import CDense
from .MPSHead import REAL_NODE_EPS

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype
MAX_NODE = 1 << 20  # entries of one node tensor


def isometry_init(cplx):
    """Random isometry (out, n): orthonormal rows if out <= n, else orthonormal columns. Complex as (..., 2)."""

    def init(key, shape, dtype):
        out, n = shape[:2]
        k1, k2 = jax.random.split(key)
        g = jax.random.normal(k1, (max(out, n), min(out, n)), jnp.float64)
        if cplx:
            g = g + 1j * jax.random.normal(k2, g.shape, jnp.float64)
        q = jnp.linalg.qr(g)[0]
        m = q.T if out <= n else q
        return (jnp.stack([m.real, m.imag], -1) if cplx else m).astype(dtype)

    return init


def node_apply(node, vecs):
    """node (out, d^n), legs in the order of vecs (C order); vecs: n arrays (..., d) -> (..., out)."""
    d = vecs[0].shape[-1]
    r = jnp.einsum("okd,...d->...ok", node.reshape(node.shape[0], -1, d), vecs[-1])
    for v in reversed(vecs[:-1]):
        r = jnp.einsum("...okd,...d->...ok", r.reshape(*r.shape[:-1], -1, d), v)
    return r[..., 0]


def rescale(v):
    # v = exp(log_s) * v', max|v'| = 1 over the last axis; exact for any s, so s carries no gradient
    s = jax.lax.stop_gradient(jnp.max(jnp.abs(v), axis=-1))
    return v / s[..., None], jnp.log(s)


def ttn_log_psi(leaves, w1, w2, w3):
    """
    leaves (B, H, W, K, d0) (K = spins per token, C order of the token), nodes w1 (d1, d0^K),
    w2 (d2, d1^W), w3 (1, d2^H) -> (B, 1) complex log of the tree summed over the H*W grid shifts.
    """
    B, H, W, K, _ = leaves.shape
    leaves, ls = rescale(leaves)
    log_c = jnp.sum(ls, axis=(1, 2, 3))  # every shift uses every leaf and token once: common factors
    u, ls = rescale(node_apply(w1, [t[..., 0, :] for t in jnp.split(leaves, K, axis=3)]))  # (B, H, W, d1)
    log_c = log_c + jnp.sum(ls, axis=(1, 2))
    U = jnp.stack([jnp.roll(u, -c, axis=2) for c in range(W)], axis=1)  # U[:, c, i, j] = u[:, i, j + c]
    r, ls = rescale(node_apply(w2, [t[..., 0, :] for t in jnp.split(U, W, axis=3)]))  # (B, W, H, d2)
    log_t = jnp.sum(ls, axis=2)  # (B, W): every row shift uses every row once
    R = jnp.stack([jnp.roll(r, -s, axis=2) for s in range(H)], axis=2)  # R[:, c, s, i] = r[:, c, i + s]
    t = node_apply(w3, [q[..., 0, :] for q in jnp.split(R, H, axis=3)])[..., 0]  # (B, W, H)
    # sum the scaled terms, then one log (no per-term log: a zero term would give a NaN gradient)
    m = jnp.max(log_t, axis=1)
    t = jnp.exp(log_t - m[:, None])[..., None] * t
    A, S = jnp.sum(t, axis=(1, 2)), jnp.sum(jnp.abs(t), axis=(1, 2))
    return (log_c + m + jnp.log(A + 1j * REAL_NODE_EPS * S))[:, None]


class SpinTTNHead(nn.Module):
    """
    Backflow tree tensor network with one leaf per spin (see the module docstring).

    Input (B, H, W, C) token-grid features; output (B, 1) complex log psi.

    token_size: (sy, sx), spins per token (ViT2D token_size, CvT total stride)
    leaf_dim:   d0, dimension of the leaf vectors (one per spin)
    token_dim:  d1, output of the token node (sy*sx leaves -> 1)
    row_dim:    d2, output of the row node (W tokens -> 1)
    features:   h, features per spin entering the leaf map
    init_scale: std of the leaf entries K.y at init (leaf = e0 + O(init_scale); 0 = exactly flat log psi)
    real:       real tensors (real psi)
    """

    token_size: tuple[int, int]
    leaf_dim: int = 4
    token_dim: int = 8
    row_dim: int = 8
    features: int = 16
    init_scale: float = 0.05
    real: bool = False

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"SpinTTNHead needs the feature grid (B, H, W, C), got shape {x.shape}")
        B, H, W, _ = x.shape
        sy, sx = self.token_size
        K, h = sy * sx, self.features
        d0, d1, d2 = self.leaf_dim, self.token_dim, self.row_dim
        for name, size in (("token", d1 * d0**K), ("row", d2 * d1**W), ("top", d2**H)):
            if size > MAX_NODE:
                raise ValueError(f"SpinTTNHead: the {name} node has {size} > {MAX_NODE} entries on a {H}x{W} token "
                                 f"grid with tokens {sy}x{sx}; lower leaf_dim/token_dim/row_dim")
        cplx = not self.real
        cdtype = jnp.complex64 if DTYPE == jnp.float32 else jnp.complex128

        y = nn.Dense(K * h, dtype=DTYPE, param_dtype=DTYPE)(x.astype(DTYPE))
        y = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(nn.silu(y).reshape(B, H, W, K, h))
        kernel_init = nn.initializers.normal(self.init_scale / np.sqrt(h))
        e0 = lambda key, shape, dtype: jnp.zeros(shape, dtype).at[(0,) * len(shape)].set(1)  # leaf bias = e0
        if cplx:
            leaves = CDense(d0, param_dtype=DTYPE, kernel_init=kernel_init, bias_init=e0)(y)
        else:
            leaves = nn.Dense(d0, dtype=DTYPE, param_dtype=DTYPE, kernel_init=kernel_init, bias_init=e0)(y)

        def node(name, out, n_in):
            p = self.param(name, isometry_init(cplx), (out, n_in, 2) if cplx else (out, n_in), DTYPE)
            return (p[..., 0] + 1j * p[..., 1]).astype(cdtype) if cplx else p.astype(DTYPE)

        w1 = node("w_token", d1, d0**K)
        w2 = node("w_row", d2, d1**W)
        w3 = node("w_top", 1, d2**H)
        return ttn_log_psi(leaves, w1, w2, w3)
