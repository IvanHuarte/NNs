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
starts from its own state (up to O(init_scale)).

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


class MPSHead(nn.Module):
    """
    Backflow-MPS head (see the module docstring).

    Input (B, H, W, C) feature grid; output (B, 1) complex log psi.

    bond_dim:   D, bond dimension of each MPS
    n_mps:      number of MPS summed (block-diagonal MPS of bond n_mps*D)
    init_scale: std of the matrix entries T.x_r at init (A = 1 + O(init_scale))
    """

    bond_dim: int
    n_mps: int = 2
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"MPSHead needs the feature grid (B, H, W, C), got shape {x.shape}")
        B, H, W, C = x.shape
        n, D = self.n_mps, self.bond_dim
        cdtype = jnp.complex64 if DTYPE == jnp.float32 else jnp.complex128
        eye = jnp.eye(D, dtype=cdtype)

        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x.astype(DTYPE))
        mats = CDense(
            n * D * D,
            param_dtype=DTYPE,
            kernel_init=nn.initializers.normal(self.init_scale / np.sqrt(C)),
        )(x)
        mats = mats.reshape(B, H, W, n, D, D) + eye

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
        R = []
        for i in range(H):
            a = [mats[:, i, j] for j in range(W)]
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
        log_psi = log_s[:, None] + ls + jnp.log(jnp.trace(M, axis1=-2, axis2=-1))  # (B, W, n)
        return logsumexp_cplx(log_psi.reshape(B, W * n), axis=1)[:, None]
