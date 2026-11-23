"""
PEPSHead: projected-entangled-pair-state readout of a feature grid (backflow PEPS).

Neural tensor network states, Fan et al., "Disentangling Tensor Network States
with Deep Neural Networks", arXiv:2603.14425: a deep network disentangles the
configuration into per-site features and a tensor network whose tensors depend
on them carries the residual correlations. The paper uses an MPS (MPSHead) and
names PEPS as the 2D upgrade; NN-generated PEPS tensors work for the Hubbard
model (NN-fPEPS, Du, Chen, Chan, arXiv:2506.08329). Here one PEPS tensor per
token of the backbone grid, with the physical index absorbed in the features as
in MPSHead:

    A_r[l, r, u, d] = delta_lr delta_ud + sum_a x_r^a T^a[l, r, u, d]   (T shared by all sites)
    psi(s) = contraction of the A_r on the H x W torus (each bond joins two neighbouring tokens)

Exact contraction with row transfer matrices,

    R_i[(u_0 .. u_{W-1}), (d_0 .. d_{W-1})] = Tr_h prod_j A_{i,j}      (D^W x D^W)
    psi = Tr R_0 R_1 ... R_{H-1}.

A translation of the grid shifts the rows (a cyclic shift of the product) and
the columns (the same permutation of the tensor factors of every R_i), so psi
is invariant under every translation of the H x W grid with a single
contraction (MPSHead sums W column offsets). Under Sequivariant the momentum
comes from z and the coset adds the intra-token translations, as for MPSHead.

Input: (B, H, W, C), the feature grid of a backbone with final_architecture=None.
Output: (B, 1) complex log psi.

Init: A = identity (a product of closed loops, psi = D^(H+W) = const) plus T.x
of std init_scale. Near T = 0 the head is, up to second order in T, a function
of pooled sums of the A_r (the MPS case is worked out in the explorer notes),
redundant with OutputHead and stuck at a saddle; init_scale ~ 1 is a generic
PEPS, meant for training together with the backbone.

Cost per sample: H * (D^(2W+3) to build R_i + D^(3W) for the product) complex
MACs, plus one C -> D^4 complex Dense per token. Exponential in W: D = 4 costs
~3e4 on a 2x2 grid (4x4 lattice, tokens 2x2) and ~1e6 on 3x3 (6x6); the 5x5
grid of 10x10 needs D <= 3.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ...toolbox import CDense

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype


class PEPSHead(nn.Module):
    """
    Backflow-PEPS head (see the module docstring).

    Input (B, H, W, C) feature grid; output (B, 1) complex log psi.

    bond_dim:   D, bond dimension of the PEPS (D^4 complex entries per tensor)
    init_scale: std of the tensor entries T.x_r at init (A = identity + O(init_scale))
    """

    bond_dim: int
    init_scale: float = 0.5

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"PEPSHead needs the feature grid (B, H, W, C), got shape {x.shape}")
        B, H, W, C = x.shape
        D = self.bond_dim
        cdtype = jnp.complex64 if DTYPE == jnp.float32 else jnp.complex128

        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x.astype(DTYPE))
        a = CDense(
            D**4,
            param_dtype=DTYPE,
            kernel_init=nn.initializers.normal(self.init_scale / np.sqrt(C)),
        )(x)
        eye = jnp.einsum("lr,ud->lrud", jnp.eye(D), jnp.eye(D)).astype(cdtype)
        a = a.reshape(B, H, W, D, D, D, D) + eye  # [l, r, u, d]

        def rescale(M, n_axes):
            # M = exp(log_s) * M', max|M'| = 1; exact for any s, so s carries no gradient
            axes = tuple(range(-n_axes, 0))
            s = jax.lax.stop_gradient(jnp.max(jnp.abs(M), axis=axes))
            return M / s.reshape(s.shape + (1,) * n_axes), jnp.log(s)

        a, log_s = rescale(a, 4)
        log_psi = jnp.sum(log_s, axis=(1, 2))  # (B,)
        P = None
        for i in range(H):
            # row MPO with open horizontal ends: M[l_0, r_j, (u_0..u_j), (d_0..d_j)]
            M = a[:, i, 0]
            for j in range(1, W):
                n = D ** (j + 1)
                M = jnp.einsum("blrUV,brsud->blsUuVd", M, a[:, i, j]).reshape(B, D, D, n, n)
            R = jnp.trace(M, axis1=1, axis2=2)  # close the horizontal loop: (B, D^W, D^W)
            P, ls = rescale(R if P is None else P @ R, 2)  # rows i-1 -> i: d of row i-1 = u of row i
            log_psi = log_psi + ls
        return (log_psi + jnp.log(jnp.trace(P, axis1=-2, axis2=-1)))[:, None]
