from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from ..toolbox import get_mask

DTYPE = jnp.float64


class CNNLiangBlock(nn.Module):
    """A simple convolutional block with a first
    It expects an input of shape x = (B,H,W,C)
    """

    M1_channels: int  # Features for convolution
    M2_channels: int  # Features for transposed convolution
    kernel: Tuple[int, int] = (3, 3)
    window_pooling: int = 2
    use_bias: bool = True
    mask: str | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        _, H, W, _ = x.shape

        mask = get_mask(self.mask) if self.mask is not None else None

        # print(f"Intro CNNLiangWorkerBlock")
        # print(f"xini: {x.shape}")

        # M1 simple convolution
        x = nn.Conv(
            features=self.M1_channels,
            kernel_size=self.kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=DTYPE,
            param_dtype=DTYPE,
            use_bias=self.use_bias,
            mask=mask,
        )(x)
        # print(f"xM1: {x.shape}")

        # x = nn.LayerNorm(
        #    dtype=DTYPE,
        #    param_dtype=DTYPE,
        # )(x)

        x = x.reshape(-1, H * W, self.M1_channels)
        # print(f"xreshape: {x.shape}")

        # Flattened max_pooling
        x = nn.max_pool(
            x,
            window_shape=(self.window_pooling,),
            strides=(self.window_pooling,),
            padding="VALID",
        )
        # print(f"xpool: {x.shape}")

        # M2 transposed convolution
        x = nn.ConvTranspose(
            features=self.M2_channels,
            kernel_size=(1,),
            strides=(self.window_pooling,),
            padding="CIRCULAR",
            dtype=DTYPE,
            param_dtype=DTYPE,
            use_bias=False,
        )(x)
        # x = nn.relu(x)
        x = x.reshape(-1, H, W, self.M2_channels)
        # x = nn.sigmoid(x)

        return x


class CNNLiang(nn.Module):
    """A CNNLiang architecture following Liang 2021 model."""

    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    "CNNLiang parameters"
    M1_channels: Tuple[int, ...]  # Features for convolution
    M2_channels: Tuple[int, ...]  # Features for transposed convolution
    kernel: Tuple[Tuple[int, ...]]  # Size of the convolutional filter
    window_pooling: int = 2
    use_bias: bool = True
    mask: str | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        n_blocks = len(self.M1_channels)
        # Entering convolutional blocks
        for i in range(n_blocks):

            x = CNNLiangBlock(
                M1_channels=self.M1_channels[i],
                M2_channels=self.M2_channels[i],
                kernel=self.kernel[i],
                window_pooling=self.window_pooling,
                use_bias=self.use_bias,
                mask=self.mask,
            )(x)

        # print(f"After all blocks: {x.shape}")
        x = x.reshape(x.shape[0], -1)

        # Apply product over 3 last indices
        x = jnp.sum(x, axis=-1, keepdims=True)
        # x = jnp.mean(x)

        # print(f"After multiplying: {x.shape}")

        return x
