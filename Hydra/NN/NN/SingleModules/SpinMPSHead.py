"""
SpinMPSHead: backflow MPS with one MPS site per spin (MPSHead at spin resolution).

Fan et al., "Disentangling Tensor Network States with Deep Neural Networks",
arXiv:2603.14425 (CNN-MPS / T-MPS): the network gives one feature vector per
spin, an FFN block acts on it, and an MPS with one site per spin, whose tensors
depend on those features, gives the amplitude. ViT2D/CvT3 features live on the
token grid, one vector per token of sy x sx spins, so MPSHead has one MPS site
per token (4 sites on 4x4 with tokens 2x2, 9 on 6x6). Here every token vector
is first split into its sy*sx spins (ViT "unpatchify") by the paper's FFN,

    y_(i sy + a, j sx + b) = SiLU(W_ab . x_ij + c_ab)        (h = features per spin)

and MPSHead runs on the (H sy, W sx, h) spin grid:

    A_n = 1 + sum_a y_n^a T^a,    psi = sum_{j<n_mps} sum_c Tr prod_n A^j_{p_n + (0, c)}.

Translations: the split is the same for every token, so a translation of the
token grid is a translation by (sy, sx) of the spin grid, under which MPSHead is
invariant. psi is invariant under the token-grid translations, like OutputHead;
the coset of Sequivariant adds the intra-token ones.

Options (all off by default; the default is the MPSHead path above):
- real: real T, so A_n is real and the MPS amplitude is real (sign 0 or pi), as
  in the paper (T_i in R). With complex T the initial phase of a long chain is
  random (on 6x6 at init 0.3 it is uniform, E ~ 0), and a small init that keeps
  it ordered leaves the MPS near the identity, where it is only a pooled readout.
  Real T near 1 gives positive amplitudes at any init up to ~0.3.
- marshall: psi times the Marshall sign (-1)^(number of up spins on the sites
  with i + j odd), the sign the paper puts in the Hamiltonian. Invariant under
  translations, C4v and the spin flip on even L at Sz = 0. With real it is the
  paper's sign setup: the MPS only learns the deviations from Marshall.
- physical: the tensor is contracted with the spin, A_n = 1 + T[s_n] . y_n,
  T[+-1] = T1 +- T2 (the physical index of the paper, d = 2). The MPS features
  are [y_n, s_n y_n] after a LayerNorm.
marshall and physical need the spins: the backbone must be ViT2DSpins, whose
last sy*sx channels are the token spins (gen.py pairs them). Guard: those
channels must be +-1 on every input (features never are); otherwise the head
raises at an eager call (Hydra builds the model eagerly) and returns NaN under
jit, so a backbone without spins cannot be read as one silently.

Other differences with the paper: T is shared by all sites and summed over the
column offsets (translation invariance; the paper uses site tensors T_i, a
position encoding and C4v). A = 1 + T.y; the paper gives no init.

Input: (B, H, W, C) feature grid of a backbone with final_architecture=None and
token/stride token_size (+ sy*sx spin channels with marshall/physical).
Output: (B, 1) complex log psi.

Cost per sample: N_tok * C * sy*sx*h for the split plus ~4 N n_mps D^3 complex
MACs for the MPS (N = number of spins); real MACs with real.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .MPSHead import MPSHead, mps_chain  # mps_chain: the same contraction for real T

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype


class SpinMPSHead(nn.Module):
    """
    MPSHead on the spin grid (see the module docstring).

    Input (B, H, W, C) token-grid features; output (B, 1) complex log psi.

    token_size: (sy, sx), spins per token (ViT2D token_size, CvT total stride)
    bond_dim:   D, bond dimension of each MPS
    features:   h, features per spin entering the MPS
    n_mps:      number of MPS summed (block-diagonal MPS of bond n_mps*D)
    init_scale: std of the matrix entries T.y_n at init (A = 1 + O(init_scale))
    init_type:  MPSHead init_type, "identity" (default) or "nilpotent" (psi unchanged at init)
    real:       real T (real MPS amplitude)
    marshall:   times the Marshall sign (needs the ViT2DSpins backbone)
    physical:   A_n = 1 + T[s_n] . y_n (needs the ViT2DSpins backbone)
    """

    token_size: tuple[int, int]
    bond_dim: int
    features: int = 16
    n_mps: int = 2
    init_scale: float = 0.5
    init_type: str = "identity"
    real: bool = False
    marshall: bool = False
    physical: bool = False

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"SpinMPSHead needs the feature grid (B, H, W, C), got shape {x.shape}")
        B, H, W, _ = x.shape
        sy, sx = self.token_size
        h = self.features
        n, D = self.n_mps, self.bond_dim

        def grid(t):  # (B, H, W, sy*sx*k) token slots -> (B, H*sy, W*sx, k) spin grid
            return t.reshape(B, H, W, sy, sx, -1).transpose(0, 1, 3, 2, 4, 5).reshape(B, H * sy, W * sx, -1)

        spins = self.marshall or self.physical
        if spins:  # the last sy*sx channels are the spins (ViT2DSpins); checked, see the guard in the docstring
            raw = x[..., -sy * sx :]
            x, s = x[..., : -sy * sx], grid(raw)[..., 0]
            spins_ok = jnp.all(jnp.abs(raw) == 1)
            try:
                concrete_bad = not bool(spins_ok)
            except jax.errors.ConcretizationTypeError:  # traced (jit): the NaN below does the job
                concrete_bad = False
            if concrete_bad:
                raise ValueError("SpinMPSHead marshall/physical read the spins from the last sy*sx channels, and "
                                 "they are not +-1: the backbone must be ViT2DSpins")
        y = nn.Dense(sy * sx * h, dtype=DTYPE, param_dtype=DTYPE)(x.astype(DTYPE))
        y = grid(nn.silu(y))
        if not (spins or self.real):
            return MPSHead(bond_dim=D, n_mps=n, init_scale=self.init_scale, init_type=self.init_type)(y)

        if self.physical:  # T[s] = T1 + s T2 on the unit-normalized features
            y = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(y)
            y = jnp.concatenate([y, s[..., None].astype(y.dtype) * y], axis=-1)
        if self.real:
            if self.init_type not in ("identity", "nilpotent"):
                raise ValueError(f'SpinMPSHead init_type={self.init_type!r}, expected "identity" or "nilpotent"')
            kernel_init = nn.initializers.normal(self.init_scale / np.sqrt(y.shape[-1]))
            if self.init_type == "nilpotent":
                mask = np.tile(np.triu(np.ones((D, D)), 1).ravel(), n)
                kernel_init = lambda key, shape, dtype, f=kernel_init: (f(key, shape, dtype) * mask).astype(dtype)
            y = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(y)
            m = nn.Dense(n * D * D, dtype=DTYPE, param_dtype=DTYPE, kernel_init=kernel_init)(y)
            log_psi = mps_chain(m.reshape(*y.shape[:3], n, D, D) + jnp.eye(D, dtype=m.dtype))
        else:
            log_psi = MPSHead(bond_dim=D, n_mps=n, init_scale=self.init_scale, init_type=self.init_type)(y)
        if self.marshall:
            odd = np.add.outer(np.arange(H * sy), np.arange(W * sx)) % 2 == 1
            parity = jnp.sum((s > 0) & odd, axis=(1, 2)) % 2
            log_psi = log_psi + (1j * jnp.pi * parity[:, None]).astype(log_psi.dtype)
        if spins:
            log_psi = jnp.where(spins_ok, log_psi, jnp.nan)
        return log_psi
