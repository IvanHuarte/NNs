from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


def get_characters(
    lattice_size: Tuple[int, int], irrep: Tuple[int, int]
) -> jnp.ndarray:
    """
    Get the characters of the irrep for the given input x
    """
    n, m = jnp.unravel_index(
        jnp.arange(jnp.prod(jnp.array(lattice_size))), lattice_size
    )

    characters = jnp.exp(
        2j * jnp.pi * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    )

    return characters[None, :, None]


class Sequivariant(nn.Module):
    """
    Flax module to project an equivariant module to an irrep
    """

    lattice_size: Tuple[int, int]

    Seq: Tuple[nn.Module, ...]
    irrep: Tuple[int, int] = (0, 0)

    def setup(self):

        self.backbone = self.Seq[:-1]
        self.final_head = self.Seq[-1]
        self.irrep = self.irrep
        self.lattice_size = self.lattice_size
        self.chars = get_characters(self.lattice_size, self.irrep)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for module in self.backbone:
            x = module(x)
        # x = (B, N, C)
        log_psi = self.final_head(x)
        # log_psi = (B, 1)

        # Get Theta_K
        phase_k = jnp.angle((self.chars * x[:, :, 0]).sum(axis=1))  # (B, 1)

        log_psi_k = log_psi + 1j * phase_k

        return log_psi_k
