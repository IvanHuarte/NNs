from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ..CvTaps import CvTaps


class CvTaps_CvTaps_Worker(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings CvTaps 1"
    n_CP_blocks_list_1: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels_list_1: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads_list_1: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides_list_1: Tuple[Tuple, ...]  # Strides for each stage
    kernel_1: Tuple  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_1: Tuple

    "Module settings CvTaps 2"
    n_CP_blocks_list_2: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels_list_2: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads_list_2: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides_list_2: Tuple[Tuple, ...]   # Strides for each stage
    kernel_2: Tuple  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_2: Tuple
    phasors: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_modulus = CvTaps(
            name="modulus",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list_1,
            channels_list=self.channels_list_1,
            attn_heads_list=self.attn_heads_list_1,
            strides_list=self.strides_list_1,
            kernel=self.kernel_1,
            final_architecture=self.final_architecture_1,
        )(x)

        phase = CvTaps(
            name="phase",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list_2,
            channels_list=self.channels_list_2,
            attn_heads_list=self.attn_heads_list_2,
            strides_list=self.strides_list_2,
            kernel=self.kernel_2,
            final_architecture=self.final_architecture_2,
            phasors=self.phasors,
        )(x)

        return log_modulus + 1j * phase


class CvTaps_CvTaps_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings CvTaps 1"
    n_CP_blocks_list_1: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels_list_1: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads_list_1: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides_list_1: Tuple[Tuple, ...]  # Strides for each stage
    kernel_1: Tuple   # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_1: Tuple 

    "Module settings CvTaps 2"
    n_CP_blocks_list_2: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels_list_2: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads_list_2: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides_list_2: Tuple[Tuple, ...]   # Strides for each stage
    kernel_2: Tuple   # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_2: Tuple 
    phasors: bool = False

    "Symmetries"
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        worker = CvTaps_CvTaps_Worker(
            lattice_size=self.lattice_size,
            n_CP_blocks_list_1=self.n_CP_blocks_list_1,
            channels_list_1=self.channels_list_1,
            attn_heads_list_1=self.attn_heads_list_1,
            strides_list_1=self.strides_list_1,
            kernel_1=self.kernel_1,
            final_architecture_1=self.final_architecture_1,
            n_CP_blocks_list_2=self.n_CP_blocks_list_2,
            channels_list_2=self.channels_list_2,
            attn_heads_list_2=self.attn_heads_list_2,
            strides_list_2=self.strides_list_2,
            kernel_2=self.kernel_2,
            final_architecture_2=self.final_architecture_2,
            phasors=self.phasors,
        )

        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Ahora sí podemos concatenar
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=0)
            return res
        else:
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)
            return res


class CvTaps_CvTaps(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings CvTaps 1"
    n_CP_blocks_list_1: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels_list_1: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads_list_1: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides_list_1: Tuple[Tuple, ...]  # Strides for each stage
    kernel_1: Tuple   # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_1: Tuple 

    "Module settings CvTaps 2"
    n_CP_blocks_list_2: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    channels_list_2: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    attn_heads_list_2: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    strides_list_2: Tuple[Tuple, ...]   # Strides for each stage
    kernel_2: Tuple   # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_2: Tuple 
    phasors: bool = False

    "Symmetries"
    symm_Z2: bool = False
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        if self.symm_Z2:
            return CvTaps_CvTaps_Z2(
                lattice_size=self.lattice_size,
                n_CP_blocks_list_1=self.n_CP_blocks_list_1,
                channels_list_1=self.channels_list_1,
                attn_heads_list_1=self.attn_heads_list_1,
                strides_list_1=self.strides_list_1,
                kernel_1=self.kernel_1,
                final_architecture_1=self.final_architecture_1,
                n_CP_blocks_list_2=self.n_CP_blocks_list_2,
                channels_list_2=self.channels_list_2,
                attn_heads_list_2=self.attn_heads_list_2,
                strides_list_2=self.strides_list_2,
                kernel_2=self.kernel_2,
                final_architecture_2=self.final_architecture_2,
                phasors=self.phasors,
                trivial_Z2=self.trivial_Z2,
            )(x)
        else:
            return CvTaps_CvTaps_Worker(
                lattice_size=self.lattice_size,
                n_CP_blocks_list_1=self.n_CP_blocks_list_1,
                channels_list_1=self.channels_list_1,
                attn_heads_list_1=self.attn_heads_list_1,
                strides_list_1=self.strides_list_1,
                kernel_1=self.kernel_1,
                final_architecture_1=self.final_architecture_1,
                n_CP_blocks_list_2=self.n_CP_blocks_list_2,
                channels_list_2=self.channels_list_2,
                attn_heads_list_2=self.attn_heads_list_2,
                strides_list_2=self.strides_list_2,
                kernel_2=self.kernel_2,
                final_architecture_2=self.final_architecture_2,
                phasors=self.phasors
            )(x)
