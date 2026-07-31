from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from NN_module.utils import _angle

def get_characters(
    lattice_size: Tuple[int, int], irrep: Tuple[int, int]
) -> jnp.ndarray:
    """
    Get the characters of the irrep for the given input x
    """
    xs = jnp.arange(0, lattice_size[1])
    ys = jnp.arange(0, lattice_size[0])

    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    indexes = jnp.stack([yy, xx], axis=-1)

    n = indexes[..., 0].flatten()
    m = indexes[..., 1].flatten()

    characters = jnp.exp(
        2.0j * jnp.pi * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    ).reshape(1, *lattice_size)

    # characters = 2.0j * jnp.pi * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    # characters = characters.reshape(1, *lattice_size)
    # print(characters)

    return characters


class Sequivariant(nn.Module):
    """
    Flax module to project an equivariant module to an irrep
    """

    SeqV: Tuple[nn.Module, ...]
    irrep: Tuple[int, int] = (0, 0)
    eps: float = 1e-6

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for module in self.SeqV[:-1]:
            x = module(x)

        _, H, W, _ = x.shape

        characters = get_characters((H, W), self.irrep)  # (B, H, W)

<<<<<<< HEAD

        # Get Theta_K
        z = (characters * x[...,0].astype(jnp.complex128)).sum(axis=(1,2))  # phase_k = (B, 1)
        phase_k = _angle(z)
        # phase_k = jnp.arctan2(jnp.imag(z), jnp.real(z) + self.eps)[:, None]

        # Get Theta_K
=======
        # Get Theta_K
        z = (characters * x[...,0].astype(jnp.complex128)).sum(axis=(1,2))
        phase_k = jnp.arctan2(jnp.imag(z), jnp.real(z) + self.eps)[:, None]



        # Get Theta_K
>>>>>>> refs/remotes/origin/dev
        # eps = 1e-6
        # z_eps = z + eps + 1e-10j
        # phase_k = jnp.angle(z_eps)[:, None]


<<<<<<< HEAD
=======
        # Get Theta_K
        # eps = 1e-6
        # abs_z = jnp.sqrt(jnp.real(z)**2 + jnp.imag(z)**2) + eps
        # cos_phi = jnp.real(z) / abs_z
        # sin_phi = jnp.imag(z) / abs_z
        # phase_k = jnp.arctan2(sin_phi, cos_phi)[:, None]
        
>>>>>>> refs/remotes/origin/dev

        log_psi = self.SeqV[-1](x)  # log_psi = (B, 1)

        log_psi_k = log_psi + 1j * phase_k  # log_psi_k = (B, 1)
        # log_psi_k = log_psi_k.real + 1.0j * ((log_psi_k.imag + jnp.pi) % (2 * jnp.pi) - jnp.pi)

        # jax.debug.print("x: \n{}",x[0].reshape(-1))
        # jax.debug.print("characters: \n{}",characters[0].reshape(-1))
        # jax.debug.print("phase_k: \n{}",phase_k[0].reshape(-1))
        # jax.debug.print("log_psi: \n{}",log_psi[0].reshape(-1))
        # jax.debug.print("log_psi_k: \n{}",log_psi_k[0].reshape(-1))

        return log_psi_k
