"""
SpinBTTNHead: SpinTTNHead with a binary tree above the token nodes (backflow binary TTN).

SpinTTNHead joins the token vectors with a row node w2 (d2 x d1^W) and a top node w3 (d2^H), exponential in
the width and height of the token grid: fine on 6x6 (tokens 2x2 -> 3x3 grid), but w2 alone has 262k complex
entries on 10x10 (5x5 grid) and ~2M on 12x12, and no bond above 8 fits. Here the leaves and the token node
are those of SpinTTNHead (same functions, same parameter names and init), and above the tokens sits a binary
tree, the standard TTN (Shi, Duan, Vidal, PRA 74, 022320 (2006)), whose cost is O(N chi^3):

    leaves   v_n = e0 + K . LN(SiLU(W_ab . x_t + c_ab))        one per spin, dimension d0
    token    u_t = w1 (v_t1 x ... v_t(sy*sx))                  w1: d1 x d0^(sy*sx), shared by the tokens
    node     z_n = T_n (z_left x z_right), z_t = u_t            T_n: chi_n x chi_left chi_right, one per node
    psi      = sum over the H*W cyclic shifts of the token grid of the root amplitude (chi_root = 1)

Tree: recursive bisection of the H x W token grid. A block is split along its longer side (columns on a tie)
into a floor half (top/left) and a ceil half, down to single tokens: N_t - 1 internal nodes for N_t = H W
tokens. chi of a token is d1, of an internal node min(bond_dim, chi_left chi_right), of the root 1. Every
node has its own tensor. The level of a node is its height (1 = both children are tokens); the nodes of a
level with the same shapes are contracted together by one einsum over (node, batch, shift).

6x6 lattice, tokens 2x2 -> 3x3 token grid (t_ij = token (i, j); [h x w] = block of the node, L = level):

    [3x3] root, L4: (1, chi, chi)
     |-- [3x1] col 0, L2
     |    |-- t00
     |    `-- [2x1] L1: t10, t20
     `-- [3x2] cols 1-2, L3
          |-- [1x2] L1: t01, t02
          `-- [2x2] L2
               |-- [2x1] L1: t11, t21
               `-- [2x1] L1: t12, t22

    level 1: 4 nodes (chi1, d1, d1), chi1 = min(bond, d1^2); level 2: [3x1] (chi, d1, chi1) and
    [2x2] (chi, chi1, chi1); level 3: [3x2]; level 4: the root.

10x10 lattice -> 5x5 grid: 24 internal nodes in 6 levels (12, 5, 3, 2, 1, 1 nodes). The root splits the
columns 2 | 3, each half splits the rows 2 | 3, and so on; the [3x3] block is the 6x6 tree above.
12x12 -> 6x6 grid: 35 nodes in 6 levels (16, 8, 4, 4, 2, 1).

Translations: the leaves and token nodes are the same for every token (equivariant), and psi sums the tree
over the H*W cyclic shifts of the token grid, so it is exactly invariant under the token-grid translations,
like SpinTTNHead. The coset of Sequivariant adds the intra-token translations; Z2 and C4v act outside.

Numerics (as SpinTTNHead): leaves, tokens and every internal node but the root are rescaled by their max
(stop_gradient, exact) and the log scales are summed (per shift for the tree nodes); the H*W root terms are
added as numbers and one log is taken (log_sum_shifts). A node is one GEMM on the outer product of its
children, T_n (z_left x z_right); the outer products are recomputed in the backward pass (jax.checkpoint), not
kept from the forward pass (CPU, 10x10, bond 32: 5-8 % less temporary memory, same time).

Init: leaves ~e0 (init_scale) and token node a random isometry, as SpinTTNHead, so every token vector is
~w1[:, 0] and log psi is flat over configurations whatever the tree tensors (exactly flat at init_scale 0).
Tree nodes, init_type:
  "identity"  T_n[0, 0, 0] = 1 plus init_scale times a standard normal on every entry (real and imaginary
              parts): the tree starts as a product of one component per token, plus O(init_scale).
  "isometry"  random isometries (QR) of shape (chi_n, chi_left chi_right), like w2 and w3 of SpinTTNHead:
              generic environments for every node.
Both give the same spread of log psi at init (it comes from the leaves). But "identity" reads the component 0
of each token vector, whose size is set by the random w1: on 6x6 (bond 16, init_scale 0.05) the variance of the
log-derivatives per node falls 16x from level 1 to the root for one seed (upper levels almost inert, as the
identity of papers.md section 0) and is ~100x larger at every level for another, while "isometry" keeps every
level active (root 3-4x level 1) for every seed tried.

Input: (B, H, W, C) feature grid of a backbone with final_architecture=None and token/stride token_size.
Output: (B, 1) complex log psi (real tensors: real psi, phase 0 or pi).

Cost per sample (complex MACs): leaves and tokens as SpinTTNHead, plus the tree,
H W sum_n chi_n chi_left chi_right <= H W (N_t - 1) chi^3: 72 chi^3 on 6x6 (9 shifts x 8 nodes),
600 chi^3 on 10x10 (25 x 24), 1260 chi^3 on 12x12 (36 x 35).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .SpinTTNHead import MAX_NODE, isometry_init, log_sum_shifts, node_param, rescale, spin_leaves, token_vectors

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype


def bisection_tree(H, W):
    """
    Binary tree of the H x W token grid: a block is split along its longer side (columns on a tie) into a floor
    half (top/left) and a ceil half, down to single tokens. Returns the internal nodes, children first (root
    last), as (left, right, (i0, j0, h, w)): token (i, j) has id i W + j, internal node k has id H W + k.
    """
    P, tree = H * W, []

    def split(i0, j0, h, w):
        if h * w == 1:
            return i0 * W + j0
        if w >= h:
            a = w // 2
            kids = split(i0, j0, h, a), split(i0, j0 + a, h, w - a)
        else:
            a = h // 2
            kids = split(i0, j0, a, w), split(i0 + a, j0, h - a, w)
        tree.append((*kids, (i0, j0, h, w)))
        return P + len(tree) - 1

    split(0, 0, H, W)
    return tree


def tree_dims(tree, P, d1, bond_dim):
    """chi of every node id: d1 for the P tokens, min(bond_dim, chi_left chi_right) inside, 1 at the root."""
    chi = [d1] * P
    for k, (l, r, _) in enumerate(tree):
        chi.append(1 if k == len(tree) - 1 else min(bond_dim, chi[l] * chi[r]))
    return chi


def shift_index(H, W):
    """(P, P) index: token at position p = i W + j of the grid shifted by s = a W + b, ((i+a)%H) W + (j+b)%W."""
    i, j, a, b = np.ix_(range(H), range(W), range(H), range(W))
    return (((i + a) % H) * W + (j + b) % W).reshape(H * W, H * W)


@jax.checkpoint  # the backward pass recomputes the outer products (chi_l chi_r per node and shift)
def binary_nodes(T, L, R):
    """T (G, chi, kl kr) node tensors, children L (G, ..., kl) and R (G, ..., kr) -> (G, ..., chi), T (L x R)."""
    X = (L[..., :, None] * R[..., None, :]).reshape(*L.shape[:-1], -1)
    return jnp.einsum("g...k,gak->g...a", X, T)


def btree_amplitude(U, nodes, tree):
    """
    U (P, ..., d1): token vectors at the P positions of the grid (any batch dims), nodes: tensors
    (chi_n, chi_left chi_right) in the order of tree -> t, log_t (...): the root amplitude is exp(log_t) t.
    Level by level: the nodes of a level with the same shapes are one einsum.
    """
    P = U.shape[0]
    vals, chi, height = dict(enumerate(U)), [U.shape[-1]] * P, [0] * P
    for (l, r, _), T in zip(tree, nodes):
        chi.append(T.shape[0])
        height.append(1 + max(height[l], height[r]))
    levels = {}
    for k, (l, r, _) in enumerate(tree):
        levels.setdefault((height[P + k], chi[P + k], chi[l], chi[r]), []).append(k)
    log_t = jnp.zeros(U.shape[1:-1], jnp.finfo(U.dtype).dtype)
    for key in sorted(levels):
        ks = levels[key]
        v = binary_nodes(jnp.stack([nodes[k] for k in ks]), jnp.stack([vals[tree[k][0]] for k in ks]),
                         jnp.stack([vals[tree[k][1]] for k in ks]))  # (G, ..., chi)
        if ks[-1] == len(tree) - 1:  # the root (alone in the top level): scalar, not rescaled
            return v[0, ..., 0], log_t
        v, ls = rescale(v)
        log_t = log_t + jnp.sum(ls, axis=0)
        vals.update((P + k, v[g]) for g, k in enumerate(ks))


def btree_log_psi(leaves, w1, nodes, tree):
    """
    leaves (B, H, W, K, d0), token node w1 (d1, d0^K), tree node tensors (chi_n, chi_left chi_right) in the
    order of tree = bisection_tree(H, W) -> (B, 1) complex log of the tree summed over the H*W grid shifts.
    """
    B, H, W = leaves.shape[:3]
    u, log_c = token_vectors(leaves, w1)  # (B, H, W, d1), (B,)
    U = jnp.moveaxis(u.reshape(B, H * W, -1)[:, shift_index(H, W)], 1, 0)  # (P, B, S, d1): position p, shift s
    t, log_t = btree_amplitude(U, nodes, tree)  # (B, S)
    return log_sum_shifts(log_c, log_t, t)


def near_identity_init(scale):
    """T[0, 0, ...] = 1 plus scale times a standard normal on every entry (both parts if complex)."""

    def init(key, shape, dtype):
        noise = scale * jax.random.normal(key, shape, jnp.float64)
        return noise.at[(0,) * len(shape)].add(1).astype(dtype)

    return init


class SpinBTTNHead(nn.Module):
    """
    Backflow binary tree tensor network with one leaf per spin (see the module docstring).

    Input (B, H, W, C) token-grid features (H W >= 2); output (B, 1) complex log psi.

    token_size: (sy, sx), spins per token (ViT2D token_size, CvT total stride)
    leaf_dim:   d0, dimension of the leaf vectors (one per spin)
    token_dim:  d1, output of the token node (sy*sx leaves -> 1), bond of the tree leaves
    bond_dim:   chi, largest bond of the binary tree (chi_n = min(bond_dim, chi_left chi_right))
    features:   h, features per spin entering the leaf map
    init_scale: std of the leaf entries K.y (and of the tree-node noise of "identity") at init; 0 = flat log psi
    real:       real tensors (real psi)
    init_type:  init of the tree nodes, "identity" (T_n[0, 0, 0] = 1 + noise) or "isometry" (random isometries)
    """

    token_size: tuple[int, int]
    leaf_dim: int = 4
    token_dim: int = 8
    bond_dim: int = 16
    features: int = 16
    init_scale: float = 0.05
    real: bool = False
    init_type: str = "identity"

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"SpinBTTNHead needs the feature grid (B, H, W, C), got shape {x.shape}")
        B, H, W, _ = x.shape
        sy, sx = self.token_size
        K, P = sy * sx, H * W
        d0, d1 = self.leaf_dim, self.token_dim
        if P < 2:
            raise ValueError(f"SpinBTTNHead needs at least 2 tokens, got a {H}x{W} token grid")
        tree = bisection_tree(H, W)
        chi = tree_dims(tree, P, d1, self.bond_dim)
        sizes = [("token", d1 * d0**K)] + [(f"[{h}x{w}]", chi[P + k] * chi[l] * chi[r])
                                           for k, (l, r, (_, _, h, w)) in enumerate(tree)]
        for name, size in sizes:
            if size > MAX_NODE:
                raise ValueError(f"SpinBTTNHead: the {name} node has {size} > {MAX_NODE} entries on a {H}x{W} token "
                                 f"grid with tokens {sy}x{sx}; lower leaf_dim/token_dim/bond_dim")
        inits = {"identity": near_identity_init(self.init_scale), "isometry": isometry_init(not self.real)}
        if self.init_type not in inits:
            raise ValueError(f"SpinBTTNHead init_type={self.init_type!r}, expected one of {list(inits)}")
        leaves = spin_leaves(x, self.token_size, d0, self.features, self.init_scale, self.real, DTYPE)
        w1 = node_param(self, "w_token", isometry_init(not self.real), (d1, d0**K), self.real, DTYPE)
        nodes = [node_param(self, f"w_r{i0}c{j0}_{h}x{w}", inits[self.init_type], (chi[P + k], chi[l] * chi[r]),
                            self.real, DTYPE)
                 for k, (l, r, (i0, j0, h, w)) in enumerate(tree)]
        return btree_log_psi(leaves, w1, nodes, tree)
