import flax.linen as nn
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple
from NN_module.models.SingleModels.ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64


def get_mask(name: bool = "Triangular") -> jnp.ndarray:

    if name == "Triangular":
        return jnp.array(
            [  # Mascara para red triangular
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ]
        )
    if name == "Square":
        return jnp.array(
            [  # Mascara para red cuadrada
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )
    else:
        return jnp.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )


class glu_phasor(nn.Module):
    """Transforms input into phasors, do pooling in each channel,
    applies the GLU activation function and sum over phasors.

    Args:
        x: Input array of shape (N_batch, N_spins, N_channels).

    Returns:
        A 1D array (N_batches, 1).
    """

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        # phasors
        if len(x.shape) > 2:
            x.mean(axis=1)

        x = nn.glu(x)

        x = jnp.exp(1j * x).sum(axis=-1)

        return jnp.angle(x)


class two_heads(nn.Module):
    """
    Two heads MLP for complex output.
    Args:
        x: Input array of shape (N_batch, N_spins, N_channels).

    Returns:
        A 1D array (N_batches, 1).
    """

    final_architecture: Tuple = (5,)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        x = x.reshape((x.shape[0], -1))
        log_modulus = nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))
        phase = nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))

        return (log_modulus + 1j * phase).astype(jnp.complex128).squeeze()


class two_heads_sincos(nn.Module):
    """
    Two heads for complex output with prediction for sin and cos of the phase
    Args:
        x: Input array of shape (N_batch, N_spins, N_channels).

    Returns:
        A 1D array (N_batches, 1).
    """

    final_architecture: Tuple = (5,)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        x = x.reshape((x.shape[0], -1))
        log_modulus = nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))
        sin = nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))
        cos = nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))
        phase = jnp.arctan2(sin, cos)

        return (log_modulus + 1j * phase).astype(jnp.complex128).squeeze()


class two_heads_phasors(nn.Module):
    """
    Two heads for complex output using sum of phasors for the phase part
    Args:
        x: Input array of shape (N_batch, N_spins, N_channels).

    Returns:
        A 1D array (N_batches, 1).
    """

    final_architecture: Tuple = (5,)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        if len(x.shape) > 2:
            xm = x.mean(axis=1)
            xp = x.mean(axis=1)

        phase = glu_phasor()(xp)

        log_modulus = nn.Dense(1)(
            MultiLayerPerceptron(self.final_architecture)(xm)
        ).squeeze()

        return (log_modulus + 1j * phase).astype(jnp.complex128).squeeze()


class DepthPointwiseConv(nn.Module):
    """
    Depthwise pointwise convolution
    """

    channels: int
    kernel: Tuple = (3, 3)
    strides: Tuple = (1, 1)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        Ch_in = x.shape[-1]

        mask = get_mask()
        mask = jnp.broadcast_to(mask[:, :, None, None], (*mask.shape, 1, Ch_in))
        if self.kernel[1] == 1:
            mask = None

        # Depth-wise convolution (Aplica mascara adyacente a cada canal)
        x = nn.Conv(
            features=Ch_in,
            kernel_size=self.kernel,
            feature_group_count=Ch_in,
            strides=(1, 1),
            padding="CIRCULAR",
            # mask=mask,
            dtype=REAL_DTYPE,
            use_bias=False,
        )(x)
        # Normalizacion
        x = nn.LayerNorm()(x)

        # Point-wise convolution
        x = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            dtype=REAL_DTYPE,
            use_bias=False,
        )(x)

        return x


class MarshallSign(nn.Module):
    """Marshall sign for 2D square lattices

    Args:
        x: input array of shape (B,N)
    Returns:
        array of shape (B,) with the Marshall sign
    """

    lattice_size: Tuple
    radians: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        x = x.reshape(-1, *self.lattice_size)

        Lx, Ly = self.lattice_size
        xs = jnp.arange(Lx).reshape(-1, 1) + jnp.arange(Ly).reshape(1, -1)
        marshall_pattern = (-1) ** (xs % 2)
        sign = jnp.prod(jnp.where(x > 0, marshall_pattern, 1), axis=(1, 2))

        if self.radians:
            sign = jnp.where(sign < 0, jnp.pi, 0.0)

        sign = sign.reshape(-1, 1)

        return sign
