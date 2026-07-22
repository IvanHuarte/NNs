from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.typing as jt
from netket.nn.activation import log_cosh

from ..toolbox import (
    DepthPointwiseConv,
    MultiLayerPerceptron,
)

DTYPE = jnp.float64


class ConvProjectionBlock(nn.Module):

    channels: int
    n_heads: int = 1
    kernel: Tuple = (3, 3)
    strides_qkv: Tuple[Tuple, Tuple, Tuple] = ((1, 1), (2, 2), (2, 2))
    n_mlp_layers: int = 1

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging ConvProjectionBlock")
        # print(f"Input shape: {x.shape}")

        # Convolutional projection
        # x (B,H,W,Ch_in)

        B = x.shape[0]
        assert (
            self.channels % self.n_heads == 0
        ), "Channels must be divisible by the number of heads"
        head_dim = self.channels // self.n_heads

        Q = DepthPointwiseConv(
            self.channels, kernel=self.kernel, strides=self.strides_qkv[0]
        )(x)
        K = DepthPointwiseConv(
            self.channels, kernel=self.kernel, strides=self.strides_qkv[1]
        )(x)
        V = DepthPointwiseConv(
            self.channels, kernel=self.kernel, strides=self.strides_qkv[2]
        )(x)

        _, Hq, Wq, _ = (
            Q.shape
        )  # If strides_qkv[0] != (1,1) then Hq and Wq different to H and W
        _, Hk, Wk, _ = K.shape
        Nq = Hq * Wq
        Nk = Hk * Wk

        # Reshape and transpose for multi-head attention
        Q = Q.reshape((B, Nq, self.n_heads, head_dim)).transpose(
            (0, 2, 1, 3)
        )  # Q = (B, heads, Nq, head_dim)
        K = K.reshape((B, Nk, self.n_heads, head_dim)).transpose(
            (0, 2, 1, 3)
        )  # K = (B, heads, Nk, head_dim)
        V = V.reshape((B, Nk, self.n_heads, head_dim)).transpose(
            (0, 2, 1, 3)
        )  # V = (B, heads, Nk, head_dim)

        # Self-attention block
        QKt = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(
            head_dim
        )  # QKt = (B, heads, Nq, Nk)
        atten = nn.softmax(QKt, axis=-1)
        # (B, heads, Nq, head_dim) --> (B, Nq, heads, head_dim) --> (B, Hq, Wq, channels)
        attention = (
            jnp.matmul(atten, V)
            .transpose((0, 2, 1, 3))
            .reshape((B, Hq, Wq, self.channels))
        )

        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x + attention)

        # MLP
        x_ffn = x.reshape((B, Nq, self.channels))  # Reshape to (B, Hq*Wq, channels)
        x_ffn = MultiLayerPerceptron(
            layer_widths=tuple([x_ffn.shape[-1]] * self.n_mlp_layers),
        )(x_ffn)
        x_ffn = x_ffn.reshape((B, Hq, Wq, self.channels))
        # x_ffn = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x_ffn)
        # print(f"After MLP: {x_ffn.shape}")
        return x + x_ffn


class StageBlock(nn.Module):
    """
    Implementation of a stage block for CvT.
    It consists of a convolutional token embedding followed by multiple
    convolutional projection blocks. x = (B, H, W, Ch)
    Inputs:
        n_blocks: Number of convolutional projection blocks in the stage.
        channels: Number of channels in the convolutional token embedding.
        proj_channels_setup: Tuple of tuples, where each inner tuple contains the
    """

    n_CP_blocks: int  # Number of convolutional projection blocks in the stage
    channels: Tuple[int, ...]  # Number of channels in each stage
    n_heads: int  # Number of heads for each block
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Beginning Stage")
        # print(f"Input shape: {x.shape}")

        # Convolutional token embedding
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=DTYPE,
        )(x)

        # print(f"After Conv embedding: {x.shape}")
        # Convolutional projection blocks
        for _ in range(self.n_CP_blocks):
            x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x)
            x = ConvProjectionBlock(
                channels=self.channels,
                n_heads=self.n_heads,
                kernel=self.kernel,
            )(x)

        return log_cosh(x)


class CvT(nn.Module):
    """
    Convolutional Vision Transformer (CvT) implementation.
    It consists of multiple stages, each containing a convolutional token embedding
    followed by a series of convolutional projection blocks.
    Inputs:
        n_stages: Number of stages in the CvT model.
        n_blocks: Number of convolutional projection blocks in each stage.
        channels: Number of channels in the convolutional token embedding in
                        each stage.
        proj_channels_setup: Tuple of tuples, where each inner tuple contains the
                             number of channels for each convolutional projection block
                             in the stage.
        kernel: Kernel size for the convolutional operations.
    Outputs:
        x: Output tensor. If `two_heads` is True, it returns a complex output with
           modulus and phase. Otherwise, it returns a real-valued output.
    """

    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[int, ...]  # Number of channels in each stage
    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging CvT")
        # print(f"Input shape: {x.shape}")

        B = x.shape[0]

        n_stages = len(self.n_CP_blocks)
        for i in range(n_stages):

            x = StageBlock(
                n_CP_blocks=self.n_CP_blocks[i],
                channels=self.channels[i],
                n_heads=self.attn_heads[i],
                kernel=self.kernel,
            )(x)

        # To work only with this module, we distinguish between real output
        # and imaginary output (modulus + phase).
        if self.final_architecture is None:
            return x

        # Works with termination module by default

        else:
            x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)  # Mean pooling over spins

            for hi in self.final_architecture:
                x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
            return x
