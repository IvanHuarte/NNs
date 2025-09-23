from typing import Callable, Sequence, Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from .CvT3 import CvT3
from .Phase import CNNPhasor

def complex_logsumexp(zs, axis=0):
    """
    Calcula log(sum(exp(zs))) para zs complejos, de manera estable.
    """
    m = jnp.max(jnp.real(zs), axis=axis, keepdims=True)
    scaled = jnp.exp(zs - m)
    S = jnp.sum(scaled, axis=axis)
    return jnp.log(S) + jnp.squeeze(m, axis=axis)

class SplitTraining_CvT3_CNNPhasor_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]


    "Module settings CvT3"
    n_CP_blocks_list: Tuple[int, ...]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list: Tuple[int, ...]  # Number of channels in the convolutional token embedding.
    CP_channels_list: Tuple[int, ...]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list: Tuple[int, ...]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple = (5,)

    "Module settings CNNPhasor"
    cnnph_channels: int = 64


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
            final_architecture=self.final_architecture

        )(x)

        phase = CNNPhasor(
            name="phase", 
            lattice_size=self.lattice_size,
            channels=self.cnnph_channels
            )(x)

        return (log_modulus + 1j * phase)
    
class SplitTraining_CvT3_CNNPhasor_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings CvT3"
    n_CP_blocks_list: Tuple[int, ...]         # Number of convolutional projection blocks in each stage
    CTemb_channels_list: Tuple[int, ...]     # Number of channels in the convolutional token embedding.
    CP_channels_list: Tuple[int, ...]    # Number of channels for each convolutional projection block in each stage.
    attn_heads_list: Tuple[int, ...]         # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)               # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple = (5,)

    "Module settings CNNPhasor"
    cnnph_channels: int = 64

    "Symmetries"
    symm_Z2: bool = False 
    trivial_Z2: bool = False 

    @nn.compact
    def __call__(self, x):
        
        worker = SplitTraining_CvT3_CNNPhasor_Worker(
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list,
            CTemb_channels_list=self.CTemb_channels_list,
            CP_channels_list=self.CP_channels_list,
            attn_heads_list=self.attn_heads_list,
            kernel=self.kernel,
            final_architecture=self.final_architecture,

            cnnph_channels=self.cnnph_channels
            )

        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Ahora sí podemos concatenar
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res, sgn = jax.nn.logsumexp(z2_stack, axis=0, return_sign=True)
            return res + sgn
        else:
            z2_stack_anti = z2_stack * jnp.array([1.0, -1.0])[:, None]
            res, sgn = jax.nn.logsumexp(z2_stack_anti, axis=0, return_sign=True)
            return res + 1j * jnp.angle(sgn)
        

class SplitTraining_CvT3_CNNPhasor(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings CvT3"
    n_CP_blocks_list: Tuple[int, ...]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list: Tuple[int, ...]  # Number of channels in the convolutional token embedding.
    CP_channels_list: Tuple[int, ...]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list: Tuple[int, ...]  # Number of heads for each convolutional projection block in each stage.
    kernel: Tuple = (3, 3)  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture: Tuple = (5,)

    "Module settings CNNPhasor"
    cnnph_channels: int = 64

    "Symmetries"
    symm_Z2: bool = True
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = SplitTraining_CvT3_CNNPhasor_Z2(
                lattice_size=self.lattice_size,

                n_CP_blocks_list=self.n_CP_blocks_list,
                CTemb_channels_list=self.CTemb_channels_list,
                CP_channels_list=self.CP_channels_list,
                attn_heads_list=self.attn_heads_list,
                kernel=self.kernel,
                final_architecture=self.final_architecture,

                cnnph_channels=self.cnnph_channels,

                trivial_Z2=self.trivial_Z2
                )
        else:
            worker = SplitTraining_CvT3_CNNPhasor_Worker(
                lattice_size=self.lattice_size,
                
                n_CP_blocks_list=self.n_CP_blocks_list,
                CTemb_channels_list=self.CTemb_channels_list,
                CP_channels_list=self.CP_channels_list,
                attn_heads_list=self.attn_heads_list,
                kernel=self.kernel,
                final_architecture=self.final_architecture,

                cnnph_channels=self.cnnph_channels,

            )
        
        x = worker(x)
        #jax.debug.print("x_out: {}", x)
        return x