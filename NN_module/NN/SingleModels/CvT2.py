import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple
from netket.nn.activation import log_cosh
from ..toolbox import (
    MultiLayerPerceptron,
    DepthPointwiseConv,
    two_heads,
    two_heads_phasors,
    glu_phasor,
    get_mask,
)

REAL_DTYPE = jnp.float64


class ConvProjectionBlock(nn.Module):

    channels: int
    n_heads: int = 1
    kernel: Tuple = (3, 3)
    strides_qkv: Tuple[Tuple, Tuple, Tuple] = ((1, 1), (1, 1), (1, 1))
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

        Q = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=REAL_DTYPE,
        )(x)
        K = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=REAL_DTYPE,
        )(x)
        V = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=REAL_DTYPE,
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

        x = nn.LayerNorm(param_dtype=REAL_DTYPE)(x + attention)

        # FFN block
        x_ffn = x.reshape((B, Nq, self.channels))  # Reshape to (B, Hq*Wq, channels)
        x_ffn = MultiLayerPerceptron(
            layer_widths=tuple([4 * self.channels] * self.n_mlp_layers),
            activation_function=nn.gelu,
        )(x_ffn)
        x_ffn = nn.Dense(self.channels, param_dtype=REAL_DTYPE)(x_ffn)
        x_ffn = x_ffn.reshape((B, Hq, Wq, self.channels))
        x_ffn = nn.LayerNorm(param_dtype=REAL_DTYPE)(x_ffn)
        # print(f"After MLP: {x_ffn.shape}")
        return x + x_ffn


class StageBlock(nn.Module):
    """
    Implementation of a stage block for CvT2.
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
            dtype=REAL_DTYPE,
        )(x)

        # print(f"After Conv embedding: {x.shape}")
        # Convolutional projection blocks
        for _ in range(self.n_CP_blocks):
            x = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            x = ConvProjectionBlock(
                channels=self.channels,
                n_heads=self.n_heads,
                kernel=self.kernel,
            )(x)

        return x


class CvT2Worker(nn.Module):
    """
    Convolutional Vision Transformer (CvT2) implementation.
    It consists of multiple stages, each containing a convolutional token embedding
    followed by a series of convolutional projection blocks.
    Inputs:
        n_stages: Number of stages in the CvT2 model.
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
    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False  # If True, apply GLU phasor activation before the final MLP

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging CvT2")
        # print(f"Input shape: {x.shape}")

        B = x.shape[0]
        x = x.reshape((B, *self.lattice_size, -1))

        n_stages = len(self.n_CP_blocks)
        for i in range(n_stages):

            x = StageBlock(
                n_CP_blocks=self.n_CP_blocks[i],
                channels=self.channels[i],
                n_heads=self.attn_heads[i],
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
                x = nn.Dense(1, dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(
                    MultiLayerPerceptron(self.final_architecture)(x)
                )

                return x


class CvT2_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    trivial_Z2: bool = (
        True  # If True, the wavefunction is even under global Z2 transformation
    )

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        worker = CvT2Worker(
            lattice_size=self.lattice_size,
            n_CP_blocks=self.n_CP_blocks,
            channels=self.channels,
            attn_heads=self.attn_heads,
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
            return jax.nn.logsumexp(z2_stack, axis=0, keepdims=False)
        else:
            b = jnp.asarray([1.0, -1.0])[:, None]  # shape (2,1)
            return jax.nn.logsumexp(z2_stack, b=b, axis=0, keepdims=False)


class CvT2(nn.Module):

    lattice_size: Tuple[int, int]

    n_CP_blocks: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.

    attn_heads: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple | None = None

    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    phasors: bool = False

    symm_Z2: bool = (
        False  # If True, the wavefunction is even under global Z2 transformation
    )
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        if self.symm_Z2:
            worker = CvT2_Z2(
                lattice_size=self.lattice_size,
                n_CP_blocks=self.n_CP_blocks,
                channels=self.channels,
                attn_heads=self.attn_heads,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                trivial_Z2=self.trivial_Z2,
                phasors=self.phasors,
            )
        else:
            worker = CvT2Worker(
                lattice_size=self.lattice_size,
                n_CP_blocks=self.n_CP_blocks,
                channels=self.channels,
                attn_heads=self.attn_heads,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
            )

        output_x = worker(x)
        return output_x
