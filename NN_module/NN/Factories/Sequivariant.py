from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


def get_characters(
    lattice_size: Tuple[int, int], irrep: Tuple[int, int]
) -> jnp.ndarray:
    """
    Get the characters of the irrep for the given input x
    """
    ys = jnp.arange(0, lattice_size[0])
    xs = jnp.arange(0, lattice_size[1])

    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    indexes = jnp.stack([yy, xx], axis=-1)

    n = indexes[..., 0].flatten()
    m = indexes[..., 1].flatten()

    characters = jnp.exp(
        2j * jnp.pi * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    ).reshape(1, *lattice_size)
    print(characters)

    return characters


class Sequivariant(nn.Module):
    """
    Flax module to project an equivariant module to an irrep
    """

    SeqV: Tuple[nn.Module, ...]
    irrep: Tuple[int, int] = (0, 0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for module in self.SeqV[:-1]:
            x = module(x)

        _, H, W, _ = x.shape

        characters = get_characters((H, W), self.irrep)  # (B, H, W)

        # Get Theta_K
        phase_k = jnp.angle((characters * x[:, :, :, 0]).sum(axis=(1, 2)))[
            :, None
        ]  # phase_k = (B, 1)

        log_psi = self.SeqV[-1](x)  # log_psi = (B, 1)

        log_psi_k = log_psi + 1j * phase_k  # log_psi_k = (B, 1)

        return log_psi_k
