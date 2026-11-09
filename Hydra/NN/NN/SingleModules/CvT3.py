import flax.linen as nn
import jax.numpy as jnp
import jax.typing as jt

from ...toolbox import (
    DepthPointwiseConv,
    MultiLayerPerceptron,
)

DTYPE = jnp.float64

# Activations of the FFN selectable from the JSON config (CvT3 "ffn_act";
# a key containing "activation" would be turned into callables by
# NN.utils.preprocess_setup)
ACTIVATIONS = {
    "swish": nn.swish,
    "gelu": nn.gelu,
    "relu": nn.relu,
    "elu": nn.elu,
    "tanh": jnp.tanh,
}

# Normalization schemes of a transformer block (CvT3 "norm"):
#   "mixed" (original): x = LN(x); a = Attn(x); x = LN(x + a); x = x + FFN(x)
#   "pre"             : x = x + Attn(LN(x)); x = x + FFN(LN(x))
#   "post"            : x = LN(x + Attn(x)); x = LN(x + FFN(x))
NORMS = ("mixed", "pre", "post")

# Output of the FFN after the attention (CvT3 "ffn_out"):
#   "act_ln" (original): Dense(f*C)-LN-act -> Dense(C)-LN-act  (the residual
#                        branch is swish(LN(.)) >= -0.28: it cannot remove
#                        features)
#   "linear"           : Dense(f*C)-LN-act -> Dense(C)  (standard transformer FFN)
FFN_OUTS = ("act_ln", "linear")


# SAME AS CVT BUT HAS ALSO CONFIGURABLE KERNELS AND STRIDES THROUGH STAGES
#
# Notes on the original block (kept as default, checkpoints compatible):
# - the Q/K/V projections use a 3x3 depthwise kernel in every stage: the
#   CvT3 "kernel" only sets the convolutional token embedding of each stage;
# - strides_qkv is not applied (DepthPointwiseConv ignores strides), so the
#   attention sees every token and stays translation equivariant;
# - there is no output projection after the heads (attn_out_proj=False).


class ConvProjectionBlock(nn.Module):

    channels: int
    n_heads: int = 1
    kernel: tuple = (3, 3)
    strides_qkv: tuple[tuple, tuple, tuple] = ((1, 1), (2, 2), (2, 2))
    n_mlp_layers: int = 2
    ffn_act: str = "swish"
    ffn_factor: int = 2
    ffn_out: str = "act_ln"
    norm: str = "mixed"
    attn_out_proj: bool = False

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        if self.norm not in NORMS:
            raise ValueError(f"CvT3 norm={self.norm!r}, expected one of {NORMS}")
        if self.ffn_out not in FFN_OUTS:
            raise ValueError(f"CvT3 ffn_out={self.ffn_out!r}, expected one of {FFN_OUTS}")
        act = ACTIVATIONS[self.ffn_act]
        x_res = x
        if self.norm == "pre":
            x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name="LayerNorm_pre_attn")(x)
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

        if self.attn_out_proj:  # W_O mixing the heads
            attention = nn.Dense(
                self.channels, param_dtype=DTYPE, name="attn_out_proj"
            )(attention)

        if self.norm == "pre":
            x = x_res + attention
            x_in = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name="LayerNorm_pre_ffn")(x)
        else:  # mixed / post: post-norm of the attention residual
            x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x + attention)
            x_in = x

        # FFN
        C = self.channels
        x_ffn = x_in.reshape((B, Nq, C))  # Reshape to (B, Hq*Wq, channels)
        if self.ffn_out == "act_ln":
            x_ffn = MultiLayerPerceptron(
                layer_widths=(C * self.ffn_factor, C), activation_function=act
            )(x_ffn)
        else:
            x_ffn = MultiLayerPerceptron(
                layer_widths=(C * self.ffn_factor,), activation_function=act
            )(x_ffn)
            x_ffn = nn.Dense(C, param_dtype=DTYPE, name="ffn_out")(x_ffn)
        x_ffn = x_ffn.reshape((B, Hq, Wq, C))

        x = x + x_ffn
        if self.norm == "post":
            x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name="LayerNorm_post_ffn")(x)
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
    channels: tuple[int, ...]  # Number of channels in each stage
    n_heads: int  # Number of heads for each block
    kernel: tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    strides: tuple = (1, 1)
    block_kw: dict | None = None  # ffn_act/ffn_factor/ffn_out/norm/attn_out_proj

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Beginning Stage")
        # print(f"Input shape: {x.shape}")

        # Convolutional token embedding
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=self.strides,
            padding="CIRCULAR",
            dtype=DTYPE,
            param_dtype=DTYPE,
        )(x)

        # print(f"After Conv embedding: {x.shape}")
        # Convolutional projection blocks
        block_kw = dict(self.block_kw or {})
        for _ in range(self.n_CP_blocks):
            if block_kw.get("norm", "mixed") == "mixed":
                x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x)
            x = ConvProjectionBlock(
                channels=self.channels,
                n_heads=self.n_heads,
                kernel=(3, 3),  # self.kernel,
                **block_kw,
            )(x)

        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x)

        return x


class CvT3(nn.Module):
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

    n_CP_blocks: tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: tuple[int, ...]  # Number of channels in each stage
    attn_heads: tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: tuple[tuple[int, ...], ...]
    strides: tuple[int, ...]
    final_architecture: tuple | None = None
    # Transformer-block options (defaults = original CvT3, see top of file)
    ffn_act: str = "swish"
    ffn_factor: int = 2
    ffn_out: str = "act_ln"
    norm: str = "mixed"
    attn_out_proj: bool = False

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging CvT")
        # print(f"Input shape: {x.shape}")

        B = x.shape[0]
        x = x.astype(DTYPE)  # spins come as float64 (see NN.precision)

        n_stages = len(self.n_CP_blocks)
        for i in range(n_stages):

            x = StageBlock(
                n_CP_blocks=self.n_CP_blocks[i],
                channels=self.channels[i],
                n_heads=self.attn_heads[i],
                kernel=self.kernel[i],
                strides=self.strides[i],
                block_kw=dict(
                    ffn_act=self.ffn_act,
                    ffn_factor=self.ffn_factor,
                    ffn_out=self.ffn_out,
                    norm=self.norm,
                    attn_out_proj=self.attn_out_proj,
                ),
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
