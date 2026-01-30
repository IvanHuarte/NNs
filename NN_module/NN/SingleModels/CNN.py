import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from typing import Callable, Tuple

from ..toolbox import (
    MultiLayerPerceptron,
    two_heads,
    two_heads_phasors,
    glu_phasor,
)

REAL_DTYPE = jnp.asarray(1.0).dtype


class ConvBlock(nn.Module):
    """A simple convolutional block with optional batch normalization and pooling.
    It expects an already padded input, so no additional padding is applied.
    The input shape is expected to be (batch_size, height, width, channels).
    """

    features: int
    kernel: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.swish
    use_pooling: bool = False
    pooling_strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Convolución sin padding adicional
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel,
            strides=self.strides,
            padding="CIRCULAR",
            dtype=REAL_DTYPE,
            param_dtype=REAL_DTYPE,
        )(x)

        x = nn.LayerNorm(
            dtype=REAL_DTYPE,
            param_dtype=REAL_DTYPE,
        )(x)
        x = self.activation(x)
        # print(f"Shape after ConvBlock: {x.shape}")

        if self.use_pooling:
            x = nn.avg_pool(
                x,
                window_shape=self.kernel,
                strides=self.pooling_strides,
                padding="CIRCULAR",
            )

        # print(f"Shape after pooling (if applied): {x.shape}")
        return x


class CNNWorker(nn.Module):
    """A simple 2D Convolutional Neural Network (CNN) model."""

    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    "CNN parameters"
    channels: Tuple[int, ...]  # Features for each convolutional block
    strides: Tuple[Tuple[int, int], ...]
    kernel: Tuple[int, int] = (3, 3)  # Size of the convolutional filter
    use_pooling: bool = False
    pooling_strides: Tuple[int, int] = (1, 1)
    final_architecture: Tuple[int, ...] | None = None

    "Exit modes"
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = x.reshape((-1, *self.lattice_size, 1))
        B = x.shape[0]

        # Padding periódico manual
        for i in range(len(self.channels)):

            x = ConvBlock(
                features=self.channels[i],
                strides=self.strides[i],
                kernel=self.kernel,
                use_pooling=self.use_pooling,
                pooling_strides=self.pooling_strides[i],
            )(x)

        # Works with termination module by default
        if self.final_architecture is None:
            return x

        # To work only with this module, we distinguish between real output
        # and imaginary output (modulus + phase).
        x = x.reshape(B, -1, x.shape[-1])

        if self.two_heads:

            if self.phasors:
                return two_heads_phasors(self.final_architecture)(x)

            else:
                x = x.mean(axis=1)
                x = x.reshape((B, -1))
                x = two_heads(self.final_architecture)(x)
                return x

        else:

            if self.phasors:
                return glu_phasor()(x)
            else:
                x = x.mean(axis=1)
                x = x.reshape((B, -1))
                x = nn.Dense(1, dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(
                    MultiLayerPerceptron(self.final_architecture)(x)
                )

                return x


class CNN_Z2(nn.Module):

    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    "CNN parameters"
    channels: Tuple[int, ...]  # Features for each convolutional block
    strides: Tuple[Tuple[int, int], ...]
    kernel: Tuple[int, int] = (3, 3)  # Size of the convolutional filter
    use_pooling: bool = False
    pooling_strides: Tuple[int, int] = (1, 1)
    final_architecture: Tuple[int, ...] | None = None

    "Exit modes"
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    "Symmetries"
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        worker = CNNWorker(
            lattice_size=self.lattice_size,
            channels=self.channels,
            strides=self.strides,
            kernel=self.kernel,
            use_pooling=self.use_pooling,
            pooling_strides=self.pooling_strides,
            final_architecture=self.final_architecture,
            two_heads=self.two_heads,
            phasors=self.phasors,
        )
        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Ahora sí podemos concatenar
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            return jax.nn.logsumexp(z2_stack, axis=0)
        else:
            b = jnp.asarray([1.0, -1.0])[:, None]  # shape (2,1)
            return jax.nn.logsumexp(z2_stack, b=b, axis=0)


class CNN(nn.Module):

    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    "CNN parameters"
    channels: Tuple[int, ...]  # Features for each convolutional block
    strides: Tuple[Tuple[int, int], ...]
    kernel: Tuple[int, int] = (3, 3)  # Size of the convolutional filter
    use_pooling: bool = False
    pooling_strides: Tuple[int, int] = (1, 1)
    final_architecture: Tuple[int, ...] | None = None

    "Exit modes"
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    "Symmetries"
    symm_Z2: bool = (
        False  # If True, the wavefunction is even under global Z2 transformation
    )
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        if self.symm_Z2:
            worker = CNN_Z2(
                lattice_size=self.lattice_size,
                channels=self.channels,
                strides=self.strides,
                kernel=self.kernel,
                use_pooling=self.use_pooling,
                pooling_strides=self.pooling_strides,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
                trivial_Z2=self.trivial_Z2,
            )
        else:
            worker = CNNWorker(
                lattice_size=self.lattice_size,
                channels=self.channels,
                strides=self.strides,
                kernel=self.kernel,
                use_pooling=self.use_pooling,
                pooling_strides=self.pooling_strides,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
            )

        output_x = worker(x)
        # print(f"CNN: {output_x.shape}")
        return output_x
