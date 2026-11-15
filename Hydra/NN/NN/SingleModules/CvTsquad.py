"""
CvTsquad: CvT3 with the transformer block fixed to the best option found
(the "std" variant of the Nov 2026 cluster runs).

Block (each ConvProjectionBlock), pre-norm standard transformer:
    x = x + W_O Attn(LN(x))                 depthwise-pointwise conv Q/K/V
    x = x + Dense_C( gelu(LN(Dense_4C(LN(x)))) )

Evidence (J1-J2, J2 = 0.5, f32, 4096 samples, E50 in Pauli units):
    8x8   epoch 2500: std -127.551 vs CvT3 base -127.402 (base final -127.549)
    10x10 epoch 1500: std -198.618 vs base -198.458 vs ViT e72b8 -198.483
    instability spikes 8x8 / 10x10: std 15 / 0, base 132 / 58, ffn x4 alone 101 / 90.

Defaults = the run c10_CvT3OH000std_f32: 2 stages, channels (30, 60),
heads (2, 4), blocks (1, 3), 3x3 token embeddings, stride 2 then 1.
The parameter tree equals CvT3's with norm="pre", ffn_out="linear",
ffn_act="gelu", ffn_factor=4, attn_out_proj=True: those checkpoints load here.
The Q/K/V projections use stride 1, so attention sees every token and the
network is translation equivariant (no strides_qkv).
"""

import flax.linen as nn
import jax.numpy as jnp
import jax.typing as jt

from ...toolbox import DepthPointwiseConv, MultiLayerPerceptron

DTYPE = jnp.float64  # set by NN.precision (network_dtype)
FFN_FACTOR = 4


class ConvProjectionBlock(nn.Module):

    channels: int
    n_heads: int = 1

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        B = x.shape[0]
        C = self.channels
        assert C % self.n_heads == 0, "Channels must be divisible by the number of heads"
        head_dim = C // self.n_heads

        x_res = x
        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name="LayerNorm_pre_attn")(x)

        Q = DepthPointwiseConv(C, kernel=(3, 3))(x)
        K = DepthPointwiseConv(C, kernel=(3, 3))(x)
        V = DepthPointwiseConv(C, kernel=(3, 3))(x)
        _, H, W, _ = Q.shape
        Q, K, V = (t.reshape((B, H * W, self.n_heads, head_dim)).transpose((0, 2, 1, 3)) for t in (Q, K, V))

        atten = nn.softmax(jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(head_dim), axis=-1)
        attention = jnp.matmul(atten, V).transpose((0, 2, 1, 3)).reshape((B, H, W, C))
        attention = nn.Dense(C, param_dtype=DTYPE, name="attn_out_proj")(attention)  # W_O

        x = x_res + attention
        x_in = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name="LayerNorm_pre_ffn")(x)

        x_ffn = MultiLayerPerceptron(layer_widths=(C * FFN_FACTOR,), activation_function=nn.gelu)(
            x_in.reshape((B, H * W, C))
        )
        x_ffn = nn.Dense(C, param_dtype=DTYPE, name="ffn_out")(x_ffn)
        return x + x_ffn.reshape((B, H, W, C))


class StageBlock(nn.Module):
    """Convolutional token embedding followed by n_CP_blocks transformer blocks. x = (B, H, W, Ch)."""

    n_CP_blocks: int
    channels: int
    n_heads: int
    kernel: tuple = (3, 3)  # token embedding
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=self.strides,
            padding="CIRCULAR",
            dtype=DTYPE,
            param_dtype=DTYPE,
        )(x)
        for _ in range(self.n_CP_blocks):
            x = ConvProjectionBlock(channels=self.channels, n_heads=self.n_heads)(x)
        return nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x)


class CvTsquad(nn.Module):
    """
    Convolutional Vision Transformer, one StageBlock per entry of n_CP_blocks.
    final_architecture None returns the feature map (B, H', W', C) for an
    output head; otherwise mean pooling + Dense layers of those widths.
    """

    n_CP_blocks: tuple[int, ...] = (1, 3)
    channels: tuple[int, ...] = (30, 60)
    attn_heads: tuple[int, ...] = (2, 4)
    kernel: tuple[tuple[int, ...], ...] = ((3, 3), (3, 3))
    strides: tuple[tuple[int, ...], ...] = ((2, 2), (1, 1))
    final_architecture: tuple | None = None

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        B = x.shape[0]
        x = x.astype(DTYPE)  # spins come as float64 (see NN.precision)
        for i in range(len(self.n_CP_blocks)):
            x = StageBlock(
                n_CP_blocks=self.n_CP_blocks[i],
                channels=self.channels[i],
                n_heads=self.attn_heads[i],
                kernel=tuple(self.kernel[i]),
                strides=tuple(self.strides[i]),
            )(x)
        if self.final_architecture is None:
            return x
        x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)
        for hi in self.final_architecture:
            x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
        return x
