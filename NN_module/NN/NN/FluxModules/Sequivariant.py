import flax.linen as nn
import jax.numpy as jnp

from NN_module.utils import _angle


def get_characters(
    lattice_size: tuple[int, int], irrep: tuple[int, int]
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
        2.0j
        * jnp.pi
        * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    ).reshape(1, *lattice_size)

    # characters = 2.0j * jnp.pi * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    # characters = characters.reshape(1, *lattice_size)
    # print(characters)

    return characters


class Sequivariant(nn.Module):
    """
    Flax module to project an equivariant module to an irrep
    """

    SeqV: tuple[nn.Module, ...]
    irrep: tuple[int, int]
    eps: float = 1e-10
    z_eps: float = 1e-12 + 1e-15j

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for module in self.SeqV[:-1]:
            x = module(x)

        _, H, W, _ = x.shape

        characters = get_characters((H, W), self.irrep)  # (B, H, W)

        # Get Theta_K
        z = (characters * x[..., 0].astype(jnp.complex128)).sum(axis=(1, 2))

        phase_k = _angle(z + self.z_eps)[:, None]  # phase_k = (B, 1)
        log_psi = self.SeqV[-1](x)  # log_psi = (B, 1)
        log_psi_k = log_psi + 1j * phase_k

        min_infinity = jnp.array(
            [[jnp.finfo(log_psi.dtype).min + 0j]], dtype=jnp.complex128
        )
        min_infinity = jnp.broadcast_to(min_infinity, log_psi_k.shape)

        log_psi_k = jnp.where(
            (jnp.abs(z) < self.eps)[:, None],
            min_infinity,
            log_psi_k,
        )

        # print(f"{x[..., 0]=}")
        # print(f"{characters=}")
        # print(f"char * x = \n{characters * x[..., 0]}")
        # print(f"{z=}")
        # print(f"{log_psi=}")
        # print(f"{phase_k=}")
        # print(f"{log_psi_k=}")
        # print("\n")

        return log_psi_k


class SequivariantX(nn.Module):
    """
    Flax module to project an equivariant module to an irrep
    """

    SeqVX: tuple[nn.Module, ...]
    irrep: tuple[int, int]
    eps: float = 1e-6

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        _, H, W, _ = x.shape

        characters = get_characters((H, W), self.irrep)  # (B, H, W)

        # Get Theta_K
        z = (characters * x[..., 0].astype(jnp.complex128)).sum(axis=(1, 2))
        z_safe_point = jnp.broadcast_to(jnp.array([1.0 +1.0j], dtype=z.dtype), z.shape)
        z_safe = jnp.where(jnp.abs(z) < self.eps, z_safe_point, z)
        phase_k = _angle(z_safe)[:, None]  # phase_k = (B, 1)

        # print(f"{x[..., 0]=}")
        # print(f"{characters=}")
        # print(f"char * x = \n{characters * x[..., 0]}")
        # print(f"{z=}")
        # print(f"{phase_k=}")

        for module in self.SeqVX[:-1]:
            x = module(x)

        log_psi = self.SeqVX[-1](x)  # log_psi = (B, 1)
        log_psi_k = log_psi + 1j * phase_k

        min_infinity = jnp.array(
            [[jnp.finfo(log_psi.dtype).min + 0j]], dtype=jnp.complex128
        )
        min_infinity = jnp.broadcast_to(min_infinity, log_psi_k.shape)

        log_psi_k = jnp.where(
            (jnp.abs(z) < self.eps)[:, None],
            min_infinity,
            log_psi_k,
        )

        # print(f"{log_psi=}")
        # print(f"{log_psi_k=}")
        # print("\n")

        return log_psi_k
