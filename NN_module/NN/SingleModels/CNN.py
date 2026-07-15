from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp

DTYPE = jnp.float64


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
            param_dtype=DTYPE,
        )(x)

        x = nn.LayerNorm(
            param_dtype=DTYPE,
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


class CNN(nn.Module):
    """A simple 2D Convolutional Neural Network (CNN) model."""

    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    "CNN parameters"
    channels: Tuple[int, ...]  # Features for each convolutional block
    strides: Tuple[Tuple[int, int], ...]
    kernel: Tuple[int, int] = (3, 3)  # Size of the convolutional filter
    use_pooling: bool = False
    pooling_strides: Tuple[int, int] = (1, 1)
    final_architecture: Tuple[int, ...] | None = None

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

        # To work only with this module, we distinguish between real output
        # and imaginary output (modulus + phase).
        x = x.reshape(B, -1, x.shape[-1])

        # Works with termination module by default
        if self.final_architecture is None:
            return x

        else:
            x = x.reshape(B, -1, x.shape[-1]).sum(axis=1)  # Mean pooling over spins
            for hi in self.final_architecture:
                x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
            return x
