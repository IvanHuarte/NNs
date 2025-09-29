import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64


def get_mask() -> jnp.ndarray:
    return jnp.array(
        [  # Mascara para red triangular
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
    )


def all2one_pool(x):
    return nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))


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

        x = nn.glu(x.mean(axis=1))

        x = jnp.exp(1j * x).sum(axis=-1)

        return jnp.angle(x)


class two_heads(nn.Module):
    """
    Two heads for complex output
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
    Two heads for complex output
    """

    final_architecture: Tuple = (5,)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        x = x.reshape(x.shape[0], -1, x.shape[-1])
        phase = glu_phasor()(x)

        x = x.mean(axis=1)
        log_modulus = nn.Dense(1)(
            MultiLayerPerceptron(self.final_architecture)(x)
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
        # print(f"After MLP: {x_ffn.shape}")
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
    CTemb_channels: int  # Number of channels in the convolutional token embedding
    CP_channels: int  # Number of channels for each convolutional projection block
    n_heads: int  # Number of heads for each block
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Beginning Stage")
        # print(f"Input shape: {x.shape}")

        mask = get_mask()
        mask = jnp.broadcast_to(
            mask[:, :, None, None], (*mask.shape, x.shape[-1], self.CTemb_channels)
        )
        if self.kernel[1] == 1:
            mask = None

        # Convolutional token embedding
        x = nn.Conv(
            features=self.CTemb_channels,
            kernel_size=self.kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            # mask=mask,
            dtype=REAL_DTYPE,
        )(x)

        x = nn.LayerNorm(dtype=REAL_DTYPE)(x)
        # print(f"After Conv embedding: {x.shape}")
        # Convolutional projection blocks
        for _ in range(self.n_CP_blocks):
            x = ConvProjectionBlock(
                channels=self.CP_channels,
                n_heads=self.n_heads,
                kernel=self.kernel,
            )(x)

        return log_cosh(x)


class CvTWorker(nn.Module):
    """
    Convolutional Vision Transformer (CvT) implementation.
    It consists of multiple stages, each containing a convolutional token embedding
    followed by a series of convolutional projection blocks.
    Inputs:
        n_stages: Number of stages in the CvT model.
        n_blocks: Number of convolutional projection blocks in each stage.
        CTemb_channels: Number of channels in the convolutional token embedding in
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

    n_CP_blocks_list: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple = (5,)
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    two_heads_sincos: bool = False  # The same but with sin and cos for the phase
    phasors: bool = False  # If True, apply GLU phasor activation before the final MLP

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"Begging CvT")
        # print(f"Input shape: {x.shape}")

        n_stages = len(self.n_CP_blocks_list)
        x = x.reshape((-1, *self.lattice_size, 1))

        B = x.shape[0]
        for i in range(n_stages):

            x = StageBlock(
                n_CP_blocks=self.n_CP_blocks_list[i],
                CTemb_channels=self.CTemb_channels_list[i],
                CP_channels=self.CP_channels_list[i],
                n_heads=self.attn_heads_list[i],
                kernel=self.kernel,
            )(x)

        # Phasors and glu activation if true
        if self.phasors:
            return glu_phasor()(  # Salida float (phase)
                x.reshape(x.shape[0], -1, x.shape[-1])
            )
            # return two_heads_phasors(            # Salida complex (mod + phase)
            #   self.final_architecture
            # )(x)

        x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)
        # Final MLP layer
        x = x.reshape((B, -1))
        if self.two_heads:
            return two_heads(self.final_architecture)(x)
        elif self.two_heads_sincos:
            return two_heads_sincos(self.final_architecture)(x)
        else:
            return nn.Dense(1)(
                MultiLayerPerceptron(self.final_architecture)(x)
            ).squeeze()


class CvT_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    n_CP_blocks_list: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple = (5,)
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    two_heads_sincos: bool = False  # The same but with sin and cos for the phase
    phasors: bool = False

    trivial_Z2: bool = (
        True  # If True, the wavefunction is even under global Z2 transformation
    )

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        worker = CvTWorker(
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list,
            CTemb_channels_list=self.CTemb_channels_list,
            CP_channels_list=self.CP_channels_list,
            attn_heads_list=self.attn_heads_list,
            kernel=self.kernel,
            final_architecture=self.final_architecture,
            two_heads=self.two_heads,
            two_heads_sincos=self.two_heads_sincos,
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


class CvT3(nn.Module):

    lattice_size: Tuple[int, int]

    n_CP_blocks_list: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple = (5,)
    two_heads: bool = (
        False  # If True, the output will be a complex number with modulus and phase
    )
    two_heads_sincos: bool = False  # The same but with sin and cos for the phase
    phasors: bool = False

    symm_Z2: bool = (
        False  # If True, the wavefunction is even under global Z2 transformation
    )
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        if self.symm_Z2:
            worker = CvT_Z2(
                lattice_size=self.lattice_size,
                n_CP_blocks_list=self.n_CP_blocks_list,
                CTemb_channels_list=self.CTemb_channels_list,
                CP_channels_list=self.CP_channels_list,
                attn_heads_list=self.attn_heads_list,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                two_heads_sincos=self.two_heads_sincos,
                trivial_Z2=self.trivial_Z2,
                phasors=self.phasors,
            )
        else:
            worker = CvTWorker(
                lattice_size=self.lattice_size,
                n_CP_blocks_list=self.n_CP_blocks_list,
                CTemb_channels_list=self.CTemb_channels_list,
                CP_channels_list=self.CP_channels_list,
                attn_heads_list=self.attn_heads_list,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                two_heads_sincos=self.two_heads_sincos,
                phasors=self.phasors,
            )

        output_x = worker(x)
        return output_x
