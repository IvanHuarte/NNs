import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
import flax
from typing import Tuple
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64


class CNNPhasor(nn.Module):

    lattice_size: Tuple
    channels: int

    @nn.compact
    def __call__(self, x):

        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
        x = x.reshape(-1, *self.lattice_size, 1)

        x = nn.Conv(
            features=self.channels,
            kernel_size=kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            # mask=mask,
            dtype=REAL_DTYPE,
        )(x)
        x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1], x.shape[-1])

        # phasors

        # x = nn.glu(                            # Version 1
        # jnp.exp(1j * x).mean(axis=1)
        # ).sum(axis=-1)

        x = nn.glu(x.mean(axis=1))  # Version 2
        x = jnp.exp(1j * x).sum(axis=-1)

        return jnp.angle(x)
