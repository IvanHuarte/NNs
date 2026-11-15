import flax.linen as nn
import jax.numpy as jnp
from netket.jax import logsumexp_cplx

# 1D irreps of C4v as chi(r^k s^m) = a^k b^m, with r the 90 degree rotation
# and s the reflection on the main diagonal (sigma_d):
#   A1: everything +1    A2: reflections -1
#   B1: C4 -1, sigma_v +1, sigma_d -1    B2: C4 -1, sigma_v -1, sigma_d +1
C4V_IRREPS = {"A1": (1, 1), "A2": (1, -1), "B1": (-1, -1), "B2": (-1, 1)}


def c4v_images(x):
    """The 8 images g x (B, L, L, C) -> (8, B, L, L, C), ordered as r^k s^m, m outer."""
    s = jnp.swapaxes(x, 1, 2)
    return jnp.stack([jnp.rot90(y, k, axes=(1, 2)) for y in (x, s) for k in range(4)])


def c4v_characters(irrep):
    a, b = C4V_IRREPS[irrep]
    return jnp.array([a**k * b**m for m in range(2) for k in range(4)], dtype=jnp.float32)


class C4v(nn.Module):
    """
    Explicit point-group symmetrization of a square lattice,

        psi(x) = sum_g chi(g) psi_wrapped(g x),   g in C4v (8 elements).

    The rotations are about the array center (a plaquette for even L), so
    this is an exact C4v eigenstate only if the wrapped state has momentum
    (0, 0), where every rotation centre is equivalent: use it on top of a
    translation-symmetric k=(0,0) network (Sequivariant irrep (0, 0) with
    the coset of its stride). Costs 8 forward passes (one batch).
    """

    C4vWrap: nn.Module
    irrep: str = "A1"

    def setup(self):
        self.worker = self.C4vWrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[1] != x.shape[2]:
            raise ValueError(f"C4v needs a square lattice, got {x.shape[1:3]}")
        B = x.shape[0]
        log_psi = jnp.atleast_1d(self.worker(c4v_images(x).reshape(8 * B, *x.shape[1:])))
        log_psi = log_psi.reshape(8, B, *log_psi.shape[1:])
        chi = c4v_characters(self.irrep).reshape((8,) + (1,) * (log_psi.ndim - 1))
        return logsumexp_cplx(log_psi, b=chi, axis=0)
