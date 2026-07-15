from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.typing as jt
from netket.nn.activation import log_cosh

from ..toolbox import (
    DepthPointwiseConv,
)

DTYPE = jnp.float64


class ConvMultiheadAttentionHead(nn.Module):

    channels: int
    n_heads_per_kernel: Tuple = (1,)
    kernel: Tuple[Tuple[int, int], ...] = ((3, 3),)
    strides_kv: Tuple[int, int] = (2, 2)

    def setup(self):

        total_heads = sum(self.n_heads_per_kernel)

        assert (
            self.channels % total_heads == 0
        ), "Channels must be divisible by the number of total heads"

        W_q, W_k, W_v = [], [], []
        norm_factor = []

        for i, kernel in enumerate(self.kernel):
            head_dim = self.channels // total_heads
            for _ in range(self.n_heads_per_kernel[i]):

                norm_factor.append(1.0)

                W_q.append(DepthPointwiseConv(head_dim, kernel=kernel, strides=(1, 1)))
                W_k.append(
                    DepthPointwiseConv(head_dim, kernel=kernel, strides=self.strides_kv)
                )
                W_v.append(
                    DepthPointwiseConv(head_dim, kernel=kernel, strides=self.strides_kv)
                )

        self.W_q = tuple(W_q)
        self.W_k = tuple(W_k)
        self.W_v = tuple(W_v)
        self.norm_factor = tuple(norm_factor)

    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging ConvMultiheadAttentionHead")
        # print(f"Input shape: {x.shape}")

        Q = jnp.stack(
            [
                self.W_q[i](x) / self.norm_factor[i]
                for i in range(len(self.norm_factor))
            ],
            axis=1,
        )
        K = jnp.stack(
            [
                self.W_k[i](x) / self.norm_factor[i]
                for i in range(len(self.norm_factor))
            ],
            axis=1,
        )
        V = jnp.stack(
            [
                self.W_v[i](x) / self.norm_factor[i]
                for i in range(len(self.norm_factor))
            ],
            axis=1,
        )

        Q = Q.reshape(*Q.shape[0:2], Q.shape[2] * Q.shape[3], Q.shape[4])
        K = K.reshape(*K.shape[0:2], K.shape[2] * K.shape[3], K.shape[4])
        V = V.reshape(*V.shape[0:2], V.shape[2] * V.shape[3], V.shape[4])

        return Q, K, V


class ConvProjectionBlock(nn.Module):

    channels: int
    n_heads_per_kernel: Tuple = (1,)
    kernel: Tuple[Tuple[int, int], ...] = ((3, 3),)
    strides_kv: Tuple[int, int] = (2, 2)
    n_mlp_layers: int = 1

    def setup(self):
        self.layer_norm_ini = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)
        self.layer_norm_res_1 = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)

        self.CMHA = ConvMultiheadAttentionHead(
            channels=self.channels,
            n_heads_per_kernel=self.n_heads_per_kernel,
            kernel=self.kernel,
            strides_kv=self.strides_kv,
        )

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    self.channels,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=DTYPE,
                ),
                nn.gelu,
                nn.Dense(
                    self.channels,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=DTYPE,
                ),
            ]
        )

    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        B, H, W, _ = x.shape

        x = self.layer_norm_ini(x)
        Q, K, V = self.CMHA(x)

        # Self-attention block
        QKt = jnp.matmul(Q, jnp.swapaxes(K, -2, -1))  # QKt = (B, heads, N, Nk)
        atten = nn.softmax(QKt, axis=-1)
        attention = attention = (
            jnp.matmul(atten, V)
            .transpose((0, 2, 1, 3))
            .reshape((B, H, W, self.channels))
        )  # (B, heads, N, head_dim)

        x = x + attention

        x = x + self.ff(self.layer_norm_res_1(x))  # Residual connection

        return x


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
    n_heads_per_kernel: Tuple[int, ...]  # Number of heads for each block
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)

    def setup(self):

        # Convolutional token embedding
        self.embedding = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=DTYPE,
        )

        # Convolutional projection blocks
        self.conv_proj_blocks = [
            ConvProjectionBlock(
                channels=self.channels,
                n_heads_per_kernel=self.n_heads_per_kernel,
                kernel=self.kernel,
            )
            for _ in range(self.n_CP_blocks)
        ]

    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Beginning Stage")

        x = self.embedding(x)

        # print(f"After Conv embedding: {x.shape}")
        for i in range(self.n_CP_blocks):
            x = self.conv_proj_blocks[i](x)

        return log_cosh(x)


class CvTexp(nn.Module):
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

    lattice_size: Tuple[int, int]

    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[int, ...]  # Number of channels in each stage
    attn_heads_per_kernel: Tuple[
        Tuple[Tuple[int, int], ...], ...
    ]  # Number of heads per kernel for each in each stage.
    kernels_per_stage: Tuple = (
        (3, 3),
    )  # Kernel size for the convolutional operations (must be 3x3)
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
                n_heads_per_kernel=self.attn_heads_per_kernel[i],
                kernel=self.kernels_per_stage[i],
            )(x)

        # To work only with this module, we distinguish between real output
        # and imaginary output (modulus + phase).
        x = x.reshape(B, -1, x.shape[-1])

        # Works with termination module by default
        if self.final_architecture is None:
            return x

        else:
            x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)  # Mean pooling over spins
            x = jnp.sum(log_cosh(x), axis=-1, keepdims=True)
            # for hi in self.final_architecture:
            #     x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
            return x
