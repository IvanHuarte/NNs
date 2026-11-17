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

Cost per sample: W * N * n_mps * D^3 complex MACs (N = H*W), plus one Dense.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from netket.jax import logsumexp_cplx

from ...toolbox import CDense

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype


def helical_paths(H, W):
    """(W, H*W): flat grid index of the n-th site of the row-major path shifted by c columns."""
    n = np.arange(H * W)
    return np.stack([(n // W) * W + (n % W + c) % W for c in range(W)])


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
        mats = mats.reshape(B, H * W, n, D, D) + eye

        def rescale(M):
            # product so far = exp(log_s) * M, M rescaled to max|M| = 1 (exact for any s)
            s = jax.lax.stop_gradient(jnp.max(jnp.abs(M), axis=(-2, -1)))
            return M / s[..., None, None], jnp.log(s)

        def step(carry, sites):
            M, log_s = carry
            M, ds = rescale(M @ mats[:, sites])  # (B, W, n, D, D): one more site for each column offset
            return (M, log_s + ds), None

        # The carry starts from the first site, not from constants: under shard_map (netket
        # sharding) a constant initial carry is not 'varying' and scan rejects the carry types.
        paths = jnp.asarray(helical_paths(H, W).T)  # (H*W, W): step, column offset
        (M, log_s), _ = jax.lax.scan(step, rescale(mats[:, paths[0]]), paths[1:])

        log_psi = log_s + jnp.log(jnp.trace(M, axis1=-2, axis2=-1))  # (B, W, n)
        return logsumexp_cplx(log_psi.reshape(B, W * n), axis=1)[:, None]
