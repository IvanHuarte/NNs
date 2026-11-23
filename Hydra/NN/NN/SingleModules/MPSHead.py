"""
MPSHead: matrix-product-state readout of a feature grid (backflow MPS).

Fan et al., "Disentangling Tensor Network States with Deep Neural Networks",
arXiv:2603.14425 (CNN-MPS / T-MPS, first two places of the 10x10 J1-J2
benchmark): a deep network maps the configuration to per-site features x_r and
an MPS whose matrices depend on those features carries the residual
correlations. Here

    A^j_r = 1 + sum_a x_r^a T^{j,a}          (D x D complex, T shared by all sites)
    psi(s) = sum_{j<n_mps} sum_{c<W} Tr prod_{n<N} A^j_{p_n + (0, c)}

Input: (B, H, W, C), the equivariant feature grid of a backbone with
final_architecture=None (CvT3, ViT2D, PCvT...). With a total stride or token
(sy, sx) the grid is H = L0/sy, W = L1/sx and one MPS site is one token: its
2^(sy*sx) configurations are already encoded in the C features, so the
physical index of the paper's T[C, D, D, d] is absorbed into x_r (with d = 16
it would multiply the head parameters by 16). Stride 1 is the paper's case,
one MPS site per spin.

Output: (B, 1) complex log psi, invariant under the translations of the H x W
grid, like OutputHead: under Sequivariant the momentum comes from z, and the
coset adds the intra-token translations (exact eigenstate of the full lattice,
so it can go under the C4v wrapper).

Path: row-major on the torus, p_n = (n // W, n % W). Its row change
(i, W-1) -> (i+1, 0) is a diagonal step on the torus, and a translation by
whole rows is a cyclic shift of the path, which the trace absorbs. Summing the
W column offsets c makes psi invariant under every translation of the grid
(W contractions instead of H*W).

D = 1 is a product of per-site factors, prod_r (1 + t.x_r) (CNN-like readout).
n_mps MPS of bond D are a block-diagonal MPS of bond n_mps*D (the paper uses
n_mps = 2), at n_mps^2 times less cost than a full one of that bond.

Init: T is small (init_scale), so A ~ 1 and psi ~ const. As a branch of
Transversal(OutputHead, MPSHead, operation="sum") a trained OutputHead network
starts from its own state (up to O(init_scale)). But near A = 1 only the trace of
T gets a first-order gradient (a pooled linear readout); the correlating
traceless part gets O(init_scale) (explorer/optimo.md).

init_type="nilpotent" (opt-in): T strictly upper triangular at init, so every A is
upper unitriangular and Tr prod A = D exactly: psi is unchanged at any
init_scale (warm start kept), while the lower-triangular directions get
first-order gradients proportional to init_scale. Only the init is masked;
training then moves all of T.

Cost per sample: ~4 N * n_mps * D^3 complex MACs (N = H*W), plus one Dense: the
W row products of each row come from its prefix/suffix products (~3W), and the
H rows are chained for the W offsets at once (H*W).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from netket.jax import logsumexp_cplx

from ...toolbox import CDense

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype
REAL_NODE_EPS = 1e-20  # imaginary regulator of a real MPS amplitude, relative (see mps_chain)


def mps_chain(mats):
    """
    (B, H, W, n, D, D) site matrices A (complex, or real for SpinMPSHead(real=True)) -> (B, 1) complex
    log sum_j sum_c Tr prod_n A^j along the row-major torus path shifted by c (see the module docstring).
    """
    B, H, W, n = mats.shape[:4]

    def rescale(M):
        # M = exp(log_s) * M', max|M'| = 1; exact for any s, so s carries no gradient
        s = jax.lax.stop_gradient(jnp.max(jnp.abs(M), axis=(-2, -1)))
        return M / s[..., None, None], jnp.log(s)

    def mm(a, b):
        # batched D x D product as a fused broadcast-sum: 3x faster than matmul (cuBLAS) at D = 8
        return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)

    # every path visits every site once, so the site scales are a common factor of all offsets
    mats, log_s = rescale(mats)
    log_s = jnp.sum(log_s, axis=(1, 2))  # (B, n)
    # row i with column offset c: R[i][c] = A_{i,c} ... A_{i,W-1} A_{i,0} ... A_{i,c-1}, from the
    # prefix/suffix products of the row (~3W products per row instead of W^2)
    # sites by lax.split, not mats[:, i, j]: the transpose of H*W slices is H*W full-size pads (O(N^2) backward)
    R = []
    for row in jnp.split(mats, H, axis=1):
        a = [s[:, 0, 0] for s in jnp.split(row, W, axis=2)]
        pre, suf = [a[0]], [a[-1]]
        for j in range(1, W):
            pre.append(mm(pre[-1], a[j]))  # pre[j] = a_0 ... a_j
            suf.insert(0, mm(a[W - 1 - j], suf[0]))  # suf[0] = a_{W-1-j} ... a_{W-1}
        R.append(jnp.stack([pre[-1]] + [mm(suf[c], pre[c - 1]) for c in range(1, W)], axis=1))
    # psi_c = Tr prod_i R[i][c], all offsets at once, rescaled after each row
    M, ls = rescale(R[0])  # (B, W, n, D, D)
    for i in range(1, H):
        M, ds = rescale(mm(M, R[i]))
        ls = ls + ds
    tr = jnp.trace(M, axis1=-2, axis2=-1)  # (B, W, n)
    if not jnp.iscomplexobj(tr):
        # Real matrices: sum the n*W scaled terms, then one log. A log per term + logsumexp has a NaN gradient
        # (softmax weight 0 * 1/t = inf) when a term is exactly 0, while psi is finite: a real trace sitting at
        # float32 rounding of 0 did that in E01 (optimo.md section 8). The imaginary regulator, relative to the
        # sum of |terms|, keeps log psi and its gradient finite if the whole sum is 0; elsewhere it is ~1e-20.
        c = (log_s[:, None] + ls).reshape(B, W * n)  # term scales, no gradient (stop_gradient in rescale)
        cmax = jnp.max(c, axis=1)
        t = tr.reshape(B, W * n) * jnp.exp(c - cmax[:, None])
        A, S = jnp.sum(t, axis=1), jnp.sum(jnp.abs(t), axis=1)
        return (cmax + jnp.log(A + 1j * REAL_NODE_EPS * S))[:, None]
    log_psi = log_s[:, None] + ls + jnp.log(tr.astype(jnp.result_type(tr.dtype, jnp.complex64)))
    return logsumexp_cplx(log_psi.reshape(B, W * n), axis=1)[:, None]


class MPSHead(nn.Module):
    """
    Backflow-MPS head (see the module docstring).

    Input (B, H, W, C) feature grid; output (B, 1) complex log psi.

    bond_dim:   D, bond dimension of each MPS
    n_mps:      number of MPS summed (block-diagonal MPS of bond n_mps*D)
    init_scale: std of the matrix entries T.x_r at init (A = 1 + O(init_scale))
    init_type:  "identity" (default) or "nilpotent" (T strictly upper triangular at init)
    """

    bond_dim: int
    n_mps: int = 2
    init_scale: float = 1e-2
    init_type: str = "identity"

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"MPSHead needs the feature grid (B, H, W, C), got shape {x.shape}")
        B, H, W, C = x.shape
        n, D = self.n_mps, self.bond_dim
        cdtype = jnp.complex64 if DTYPE == jnp.float32 else jnp.complex128
        eye = jnp.eye(D, dtype=cdtype)

        if self.init_type not in ("identity", "nilpotent"):
            raise ValueError(f'MPSHead init_type={self.init_type!r}, expected "identity" or "nilpotent"')
        kernel_init = nn.initializers.normal(self.init_scale / np.sqrt(C))
        if self.init_type == "nilpotent":
            mask = np.tile(np.triu(np.ones((D, D)), 1).ravel(), n)[:, None]  # (n*D*D, 1), over (C, n*D*D, 2)
            kernel_init = lambda key, shape, dtype, f=kernel_init: (f(key, shape, dtype) * mask).astype(dtype)

        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x.astype(DTYPE))
        mats = CDense(n * D * D, param_dtype=DTYPE, kernel_init=kernel_init)(x)
        mats = mats.reshape(B, H, W, n, D, D) + eye

        return mps_chain(mats)
