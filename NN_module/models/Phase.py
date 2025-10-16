import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
import flax
from typing import Tuple, Callable
from NN_module.models.CvT3 import DepthPointwiseConv

REAL_DTYPE = jnp.float64


class CNNPh(nn.Module):

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
            kernel_init=jax.nn.initializers.lecun_normal(),
        )(x)
        x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1], x.shape[-1])

        x = nn.glu(x.mean(axis=1))  # Version 2
        x = jnp.exp(1j * x).sum(axis=-1)

        return jnp.angle(x)


class EDPPh(nn.Module):
    """
    Embedded Depthwise-Pointwise convolution wit sum over Phasors
    x_inputs must be of shape (B,N,1)

    """

    lattice_size: Tuple
    channels: int
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        assert len(x.shape) == 2, f"x.shape: {x.shape}"

        N = self.lattice_size[0] * self.lattice_size[1]
        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)

        # Embedding
        x = nn.Embed(
            N,
            self.channels,
        )(x)
        x = x.reshape(-1, *self.lattice_size, self.channels)

        # Depthwise-Normalization-Pointwise
        x = DepthPointwiseConv(self.channels, kernel=kernel)(x)

        x = self.activation(x)
        x = x.reshape(-1, N, x.shape[-1])
        x = nn.Dense(self.channels)(x)

        x = nn.glu(x.mean(axis=1))  # Version JCM
        x = jnp.exp(1j * x).sum(axis=-1)
        x = jnp.angle(x)

        # jax.debug.print("EDPPh output shape: {}", x)

        return x
