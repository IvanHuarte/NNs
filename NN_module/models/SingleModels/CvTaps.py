import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple, Any
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron
from ..toolbox import (
    DepthPointwiseConv,
    two_heads,
    two_heads_phasors,
    glu_phasor,
    get_mask,
)

REAL_DTYPE = jnp.float64

###########################################################################
# APS equivariance correction functions
###########################################################################


def vmap_P_traslations(x, P_shifts):

    batched_traslations = lambda x, shift: jnp.roll(x, shift=shift, axis=(0, 1))

    return jax.vmap(batched_traslations)(x, P_shifts)


def scan_P_traslations(x, P_shifts):

    def rollit(i, x_batch):
        x_b_trasl = jnp.roll(x_batch, shift=P_shifts[i], axis=(0, 1))
        return i + 1, x_b_trasl

    _, x_trasl = jax.lax.scan(rollit, init=0, xs=x)

    return x_trasl


def polyphase_components(x, strides):

    B, H, W, C = x.shape
    Hd, Wd = (
        H // strides[0],
        W // strides[1],
    )
    xt = (
        x.reshape((B, Hd, strides[0], Wd, strides[1], C))
        .transpose((0, 1, 3, 4, 2, 5))
        .reshape((B, Hd * Wd, *strides, C))
        .transpose((0, 3, 2, 1, 4))
    )

    shape = xt.shape
    poly_comp = xt.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4])

    return poly_comp


def get_maxnorm_indices(x, strides):
    """
    This function returns the traslation indices for a batched input of shape (B, H, W, C).
    It computes the polyphase components of a grid for each batch and channel, calculates
    the L2 norm for each component and chooses the indices of the maximum value component.
    It works for both 1D and 2D inputs.
    Input:
        - x: (jnp.ArrayLike) Input data.
        - strides: (tuple) Strides for the next downsampling convolution.

    Returns:
        - x: Shifts (translations) for all batches and channels which makes the input
             traslationaly equivariant.

    """
    _, H, W, _ = x.shape
    assert (H % strides[0] == 0) & (
        W % strides[1] == 0
    ), f"`lattice_size` must be disible by `strides`. But they are {(H,W)} and {strides}"

    poly_comp = polyphase_components(x, strides)

    norm = jnp.linalg.norm(poly_comp, axis=-1)
    flat_idx = jnp.argmax(norm, axis=-1, keepdims=False)
    p, q = jnp.unravel_index(flat_idx, strides)
    anchors = jnp.array([-p, -q]).T

    return anchors


def APS_equivariance_adapter(x, strides, mode="vmap"):

    P_shifts = get_maxnorm_indices(x, strides)

    if mode == "scan":
        x = scan_P_traslations(x, P_shifts)
    elif mode == "vmap":
        x = vmap_P_traslations(x, P_shifts)
    else:
        raise ValueError(f"No such mode: {mode}")
    return x


###########################################################################
# Modules with APS equivariance correction
###########################################################################


class ConvAPS(nn.Module):

    features: int
    kernel_size: Tuple = (3, 3)
    strides: Tuple = (1, 1)
    mask: jt.ArrayLike | None = None
    feature_group_count: int = 1
    padding: str = "SAME"
    dtype: Any = REAL_DTYPE
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        if any([s > 1 for s in self.strides]):  # Downsampling APS correction
            x = APS_equivariance_adapter(x, self.strides)

        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            mask=self.mask,
            feature_group_count=self.feature_group_count,
            padding=self.padding,
            dtype=self.dtype,
            use_bias=self.use_bias,
        )(x)

        return x


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
        x = ConvAPS(
            features=Ch_in,
            kernel_size=self.kernel,
            feature_group_count=Ch_in,
            strides=self.strides,
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


class ConvProjectionBlock(nn.Module):

    channels: int
    n_heads: int = 1
    kernel: Tuple = (3, 3)
    strides_qkv: Tuple[Tuple, Tuple, Tuple] = ((1, 1), (2, 2), (2, 2))
    n_mlp_layers: int = 1

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

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

        x = nn.LayerNorm(dtype=REAL_DTYPE)(x + attention)

        # MLP
        x_ffn = x.reshape((B, Nq, self.channels))  # Reshape to (B, Hq*Wq, channels)
        x_ffn = MultiLayerPerceptron(
            layer_widths=tuple([x_ffn.shape[-1]] * self.n_mlp_layers),
        )(x_ffn)
        x_ffn = x_ffn.reshape((B, Hq, Wq, self.channels))
        x_ffn = nn.LayerNorm(dtype=REAL_DTYPE)(x_ffn)

        return x + x_ffn


class StageBlock(nn.Module):
    """
    Implementation of a stage block for CvT.
    It consists of a convolutional token embedding followed by multiple
    convolutional projection blocks. x = (B, H, W, Ch)
    Inputs:
        n_blocks: Number of convolutional projection blocks in the stage.
        CTemb_channels: Number of channels in the convolutional token embedding.
        proj_channels_setup: Tuple of tuples, where each inner tuple contains the
    """

    n_CP_blocks: int  # Number of convolutional projection blocks in the stage
    channels: int  # Number of channels in the convolutional token embedding
    n_heads: int  # Number of heads for each block
    strides: Tuple = (1, 1)  # Strides for the convolutional token embedding
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Beginning Stage")
        # print(f"Input shape: {x.shape}")

        mask = get_mask()
        mask = jnp.broadcast_to(
            mask[:, :, None, None], (*mask.shape, x.shape[-1], self.channels)
        )
        if self.kernel[1] == 1:
            mask = None

        # Convolutional token embedding
        x = ConvAPS(
            features=self.channels,
            kernel_size=self.kernel,
            strides=self.strides,
            padding="CIRCULAR",
            # mask=mask,
            dtype=REAL_DTYPE,
            use_bias=False,
        )(x)

        x = nn.LayerNorm(dtype=REAL_DTYPE)(x)
        # print(f"After Conv embedding: {x.shape}")
        # Convolutional projection blocks
        for _ in range(self.n_CP_blocks):
            x = ConvProjectionBlock(
                channels=self.channels, n_heads=self.n_heads, kernel=self.kernel
            )(x)

        return log_cosh(x)


class CvTapsWorker(nn.Module):
    """
    Convolutional Vision Transformer (CvT) implementation.
    It consists of multiple stages, each containing a convolutional token embedding
    followed by a series of convolutional projection blocks.
    Inputs:
        n_stages: Number of stages in the CvT model.
        n_blocks: Number of convolutional projection blocks in each stage.
        channels: Number of channels in the convolutional token embedding in
                        each stage.
        kernel: Kernel size for the convolutional operations.
    Outputs:
        x: Output tensor. If `two_heads` is True, it returns a complex output with
           modulus and phase. Otherwise, it returns a real-valued output.
    """

    lattice_size: Tuple[int, int]

    "CvTaps parameters"
    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides: Tuple[Tuple, ...] = (1, 1)  # Strides for each stage
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None

    "Exit modes"
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging CvT")
        # print(f"Input shape: {x.shape}")

        n_stages = len(self.n_CP_blocks)
        x = x.reshape((-1, *self.lattice_size, 1))

        B = x.shape[0]
        for i in range(n_stages):

            x = StageBlock(
                n_CP_blocks=self.n_CP_blocks[i],
                channels=self.channels[i],
                n_heads=self.attn_heads[i],
                strides=self.strides[i],
                kernel=self.kernel,
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
                return two_heads(self.final_architecture)(x)

        else:

            if self.phasors:
                return glu_phasor()(x)
            else:
                x = x.mean(axis=1)
                x = x.reshape((B, -1))
                return nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))


class CvTaps_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    "CvTaps parameters"
    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides: Tuple[Tuple, ...] = (1, 1)  # Strides for each stage
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None

    "Exit modes"
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    "Symmetries"
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        worker = CvTapsWorker(
            lattice_size=self.lattice_size,
            n_CP_blocks=self.n_CP_blocks,
            channels=self.channels,
            attn_heads=self.attn_heads,
            strides=self.strides,
            kernel=self.kernel,
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


class CvTaps(nn.Module):

    lattice_size: Tuple[int, int]

    "CvTaps parameters"
    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides: Tuple[Tuple, ...] = (1, 1)  # Strides for each stage
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None

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
            worker = CvTaps_Z2(
                lattice_size=self.lattice_size,
                n_CP_blocks=self.n_CP_blocks,
                channels=self.channels,
                attn_heads=self.attn_heads,
                strides=self.strides,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                trivial_Z2=self.trivial_Z2,
                phasors=self.phasors,
            )
        else:
            worker = CvTapsWorker(
                lattice_size=self.lattice_size,
                n_CP_blocks=self.n_CP_blocks,
                channels=self.channels,
                attn_heads=self.attn_heads,
                strides=self.strides,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
            )

        output_x = worker(x)
        return output_x
