from typing import Callable, Sequence, Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from ..CvT3 import CvT3
from ..Phase import CNNClsf


class CvT3_CNNClsf_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings CvT3"
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

    "Module settings CNNClsf"
    cnnclsf_channels: Tuple[int, ...] = (32,)
    n_classes: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_modulus = CvT3(
            name="modulus",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list,
            CTemb_channels_list=self.CTemb_channels_list,
            CP_channels_list=self.CP_channels_list,
            attn_heads_list=self.attn_heads_list,
            kernel=self.kernel,
            final_architecture=self.final_architecture,
        )(x)

        phase = CNNClsf(
            name="phase",
            lattice_size=self.lattice_size,
            channels=self.cnnclsf_channels,
            n_classes=self.n_classes,
        )(x)

        return log_modulus + 1j * phase


class CvT3_CNNClsf_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings CvT3"
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

    "Module settings CNNClsf"
    cnnclsf_channels: Tuple[int, ...] = (32,)
    n_classes: int = 2

    "Symmetries"
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        worker = CvT3_CNNClsf_Worker(
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list,
            CTemb_channels_list=self.CTemb_channels_list,
            CP_channels_list=self.CP_channels_list,
            attn_heads_list=self.attn_heads_list,
            kernel=self.kernel,
            final_architecture=self.final_architecture,
            cnnclsf_channels=self.cnnclsf_channels,
            n_classes=self.n_classes,
        )

        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Concatenamos las dos contribuciones
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=0)
            return res
        else:
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)
            return res


class CvT3_CNNClsf(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings CvT3"
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

    "Module settings CNNClsf"
    cnnclsf_channels: Tuple[int, ...] = (32,)
    n_classes: int = 2

    "Symmetries"
    symm_Z2: bool = True
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = CvT3_CNNClsf_Z2(
                lattice_size=self.lattice_size,
                n_CP_blocks_list=self.n_CP_blocks_list,
                CTemb_channels_list=self.CTemb_channels_list,
                CP_channels_list=self.CP_channels_list,
                attn_heads_list=self.attn_heads_list,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                cnnclsf_channels=self.cnnclsf_channels,
                n_classes=self.n_classes,
                trivial_Z2=self.trivial_Z2,
            )
        else:
            worker = CvT3_CNNClsf_Worker(
                lattice_size=self.lattice_size,
                n_CP_blocks_list=self.n_CP_blocks_list,
                CTemb_channels_list=self.CTemb_channels_list,
                CP_channels_list=self.CP_channels_list,
                attn_heads_list=self.attn_heads_list,
                kernel=self.kernel,
                final_architecture=self.final_architecture,
                cnnclsf_channels=self.cnnclsf_channels,
                n_classes=self.n_classes,
            )

        x = worker(x)
        # jax.debug.print("x_out: {}", x)
        return x
