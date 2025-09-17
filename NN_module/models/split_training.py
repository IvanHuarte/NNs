from typing import Callable, Sequence, Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from .MLP import BatchedMultiLayerPerceptron
from .ViT_2D import BatchedSpinViT_2D
from .CNN import CNN
from .CvT import CvT
from .CvT3 import CvT3
from .Phase import phasors_CNN

DTYPE = jnp.float64


class SplitTraining_ViT_CNN(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings ViT"
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool = False
    symm_2D_module: bool = False
    symm_Z2_module: bool = False
    trivial_Z2_module: bool = True

    "Phase settings CNN"
    block_channels: tuple = 32  # Features for each convolutional block
    kernel_size: tuple = (3, 3)  # Size of the convolutional filter
    n_ffn_layers_cnn: int = (
        1  # Number of fully connected layers after convolutional blocks
    )
    activation: Callable = nn.swish  # Activation function

    @nn.compact
    def __call__(self, batch_x: jnp.ndarray) -> jnp.ndarray:

        log_module = BatchedSpinViT_2D(
            name="modulus",
            lattice_size=self.lattice_size,
            token_size=self.token_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
            is_complex=self.is_complex,
            symm_2D=self.symm_2D_module,
            symm_Z2=self.symm_Z2_module,
            trivial_Z2=self.trivial_Z2_module,
        )(batch_x)

        phase = CNN(
            name="phase",
            lattice_size=self.lattice_size,
            block_channels=self.block_channels,
            kernel_size=self.kernel_size,
            n_ffn_layers=self.n_ffn_layers_cnn,
            activation=self.activation,
        )(batch_x)

        return log_module + 1j * phase


class SplitTraining_ViT_MLP(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings ViT"
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool = False
    symm_2D_module: bool = False
    symm_Z2_module: bool = False
    trivial_Z2_module: bool = True

    "Phase settings MLP"
    param_dtype_phase: Any = DTYPE
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1
    symm_2D_phase: bool = False
    symm_Z2_phase: bool = False
    trivial_Z2_phase: bool = True

    @nn.compact
    def __call__(self, batch_x: jnp.ndarray) -> jnp.ndarray:

        log_module = BatchedSpinViT_2D(
            name="modulus",
            lattice_size=self.lattice_size,
            token_size=self.token_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
            is_complex=self.is_complex,
            symm_2D=self.symm_2D_module,
            symm_Z2=self.symm_Z2_module,
            trivial_Z2=self.trivial_Z2_module,
        )(batch_x)

        phase = BatchedMultiLayerPerceptron(
            name="phase",
            lattice_size=self.lattice_size,
            param_dtype=self.param_dtype_phase,
            hidden_alpha=self.hidden_alpha,
            activation=self.activation,
            symm_2D=self.symm_2D_phase,
            symm_Z2=self.symm_Z2_phase,
            trivial_Z2=self.trivial_Z2_phase,
        )(batch_x)

        return log_module + 1j * phase


class SplitTraining_CvT_CNN(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings CvT"

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

    "Phase settings CNN"
    block_channels_cnn: tuple = 32  # Features for each convolutional block
    kernel_size_cnn: tuple = (3, 3)  # Size of the convolutional filter
    n_ffn_layers_cnn: int = (
        1  # Number of fully connected layers after convolutional blocks
    )
    activation: Callable = nn.swish  # Activation function

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_module = CvT(
            name="phase",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list,
            CTemb_channels_list=self.CTemb_channels_list,
            CP_channels_list=self.CP_channels_list,
            attn_heads_list=self.attn_heads_list,
            kernel=self.kernel,
            final_architecture=self.final_architecture,
        )(x)

        phase = CNN(
            name="modulus",
            lattice_size=self.lattice_size,
            block_channels=self.block_channels_cnn,
            kernel_size=self.kernel_size_cnn,
            n_ffn_layers=self.n_ffn_layers_cnn,
            activation=self.activation,
        )(x)
        # print(f"Shape after CvT and CNN: {log_module.shape}, {phase.shape}")

        return (log_module + 1j * phase).squeeze()


class SplitTraining_CvT_CvT(nn.Module):
    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings CvT 1"
    n_CP_blocks_list_1: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list_1: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list_1: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list_1: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel_1: Tuple  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_1: Tuple

    "Module settings CvT 2"
    n_CP_blocks_list_2: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list_2: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list_2: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list_2: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel_2: Tuple  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_2: Tuple

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_module = CvT(
            name="modulus",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list_1,
            CTemb_channels_list=self.CTemb_channels_list_1,
            CP_channels_list=self.CP_channels_list_1,
            attn_heads_list=self.attn_heads_list_1,
            kernel=self.kernel_1,
            final_architecture=self.final_architecture_1,
        )(x)

        phase = CvT(
            name="phase",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list_2,
            CTemb_channels_list=self.CTemb_channels_list_2,
            CP_channels_list=self.CP_channels_list_2,
            attn_heads_list=self.attn_heads_list_2,
            kernel=self.kernel_2,
            final_architecture=self.final_architecture_2,
        )(x)

        return (log_module + 1j * phase).squeeze()


class SplitTraining_CvT3_CvT3(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings CvT3 1"
    n_CP_blocks_list_1: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list_1: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list_1: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list_1: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel_1: Tuple  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_1: Tuple

    "Module settings CvT3 2"
    n_CP_blocks_list_2: Tuple[
        int, ...
    ]  # Number of convolutional projection blocks in each stage
    CTemb_channels_list_2: Tuple[
        int, ...
    ]  # Number of channels in the convolutional token embedding.
    CP_channels_list_2: Tuple[
        int, ...
    ]  # Number of channels for each convolutional projection block in each stage.
    attn_heads_list_2: Tuple[
        int, ...
    ]  # Number of heads for each convolutional projection block in each stage.
    kernel_2: Tuple  # Kernel size for the convolutional operations (must be 3x3)
    final_architecture_2: Tuple

    symm_Z2_1: bool = (
        False  # If True, the wavefunction is even under global Z2 transformation
    )
    trivial_Z2_1: bool = False
    symm_Z2_2: bool = (
        False  # If True, the wavefunction is even under global Z2 transformation
    )
    trivial_Z2_2: bool = False

    phasors: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_module = CvT3(
            name="modulus",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list_1,
            CTemb_channels_list=self.CTemb_channels_list_1,
            CP_channels_list=self.CP_channels_list_1,
            attn_heads_list=self.attn_heads_list_1,
            kernel=self.kernel_1,
            final_architecture=self.final_architecture_1,
            two_heads=False,
            two_heads_sincos=False,
            symm_Z2=self.symm_Z2_1,
            trivial_Z2=self.trivial_Z2_1,
        )(x)

        phase = CvT3(
            name="phase",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list_2,
            CTemb_channels_list=self.CTemb_channels_list_2,
            CP_channels_list=self.CP_channels_list_2,
            attn_heads_list=self.attn_heads_list_2,
            kernel=self.kernel_2,
            final_architecture=self.final_architecture_2,
            two_heads=False,
            two_heads_sincos=False,
            symm_Z2=self.symm_Z2_2,
            trivial_Z2=self.trivial_Z2_2,
            phasors=self.phasors,
        )(x)

        return (log_module + 1j * phase).squeeze()


class SplitTraining_CvT3_Phase(nn.Module):
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

    symm_Z2: bool = (
        False  # If True, the wavefunction is even under global Z2 transformation
    )
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_module = CvT3(
            name="modulus",
            lattice_size=self.lattice_size,
            n_CP_blocks_list=self.n_CP_blocks_list,
            CTemb_channels_list=self.CTemb_channels_list,
            CP_channels_list=self.CP_channels_list,
            attn_heads_list=self.attn_heads_list,
            kernel=self.kernel,
            final_architecture=self.final_architecture,
            two_heads=False,
            two_heads_sincos=False,
            symm_Z2=self.symm_Z2,
            trivial_Z2=self.trivial_Z2,
        )(x)

        # phase = four_phases_gumbel(
        #         name = 'phase'
        #     )(x)
        phase = phasors_CNN(name="phase", lattice_size=self.lattice_size)(x)

        return (log_module + 1j * phase).squeeze()
