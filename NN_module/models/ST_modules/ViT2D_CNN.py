from typing import Callable, Sequence, Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...NN_utils import traslations_2D
from ..ViT_2D import RealSpinViT
from ..CNN import CNN


class ViT2D_CNN_Worker(nn.Module):
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

    "Phase settings CNN"
    block_channels: tuple = 32  # Features for each convolutional block
    kernel_size: tuple = (3, 3)  # Size of the convolutional filter
    n_ffn_layers_cnn: int = (
        1  # Number of fully connected layers after convolutional blocks
    )
    activation: Callable = nn.swish  # Activation function

    @nn.compact
    def __call__(self, batch_x: jnp.ndarray) -> jnp.ndarray:
        # print(f"Input shape to ViT2D_CNN_Worker: {batch_x.shape}")

        token_lattice_size = (
            self.lattice_size[0] // self.token_size[0],
            self.lattice_size[1] // self.token_size[1],
        )

        log_modulus = RealSpinViT(
            name="modulus",
            token_lattice_size=token_lattice_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
        )(batch_x)

        batch_x = batch_x.reshape((-1, *self.lattice_size, 1))
        # print(f"Input shape to CNN phase: {batch_x.shape}")

        phase = CNN(
            name="phase",
            lattice_size=self.lattice_size,
            block_channels=self.block_channels,
            kernel_size=self.kernel_size,
            n_ffn_layers=self.n_ffn_layers_cnn,
            activation=self.activation,
        )(batch_x)

        return log_modulus + 1j * phase


class ViT2D_CNN_2D(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings ViT"
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool = False

    "Phase settings CNN"
    block_channels: tuple = 32  # Features for each convolutional block
    kernel_size: tuple = (3, 3)  # Size of the convolutional filter
    n_ffn_layers_cnn: int = (
        1  # Number of fully connected layers after convolutional blocks
    )
    activation: Callable = nn.swish  # Activation function

    @nn.compact
    def __call__(self, x):

        worker = ViT2D_CNN_Worker(
            lattice_size=self.lattice_size,
            token_size=self.token_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
            block_channels=self.block_channels,
            kernel_size=self.kernel_size,
            n_ffn_layers_cnn=self.n_ffn_layers_cnn,
            activation=self.activation,
        )

        # print(self.token_size, type(self.token_size))
        # print(f"x shape before traslations_2D: {x.shape}")

        # 2D traslation
        traslational_x = traslations_2D(  # shape = (token_dim, n_tokens, token_dim)
            x, size=self.lattice_size, token_size=self.token_size, memory=False
        )
        # print(f"Translational x shape: {traslational_x.shape}")

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class ViT2D_CNN_Z2(nn.Module):

    lattice_size: Tuple[int, int]

    "Module settings ViT"
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool = False

    "Phase settings CNN"
    block_channels: tuple = 32  # Features for each convolutional block
    kernel_size: tuple = (3, 3)  # Size of the convolutional filter
    n_ffn_layers_cnn: int = (
        1  # Number of fully connected layers after convolutional blocks
    )
    activation: Callable = nn.swish  # Activation function

    "Symmetries"
    symm_2D: bool = False

    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = ViT2D_CNN_2D(
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
                final_architecture=self.final_architecture,
                block_channels=self.block_channels,
                kernel_size=self.kernel_size,
                n_ffn_layers_cnn=self.n_ffn_layers_cnn,
                activation=self.activation,
            )
        else:
            worker = ViT2D_CNN_Worker(
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
                final_architecture=self.final_architecture,
                block_channels=self.block_channels,
                kernel_size=self.kernel_size,
                n_ffn_layers_cnn=self.n_ffn_layers_cnn,
                activation=self.activation,
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


class ViT2D_CNN(nn.Module):
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

    "Phase settings CNN"
    block_channels: tuple = 32  # Features for each convolutional block
    kernel_size: tuple = (3, 3)  # Size of the convolutional filter
    n_ffn_layers_cnn: int = (
        1  # Number of fully connected layers after convolutional blocks
    )
    activation: Callable = nn.swish  # Activation function

    "Symmetries"
    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):

        if self.symm_Z2:
            worker = ViT2D_CNN_Z2(
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
                final_architecture=self.final_architecture,
                block_channels=self.block_channels,
                kernel_size=self.kernel_size,
                n_ffn_layers_cnn=self.n_ffn_layers_cnn,
                activation=self.activation,
                symm_2D=self.symm_2D,
                trivial_Z2=self.trivial_Z2,
            )
        elif self.symm_2D:
            worker = ViT2D_CNN_2D(
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
                final_architecture=self.final_architecture,
                block_channels=self.block_channels,
                kernel_size=self.kernel_size,
                n_ffn_layers_cnn=self.n_ffn_layers_cnn,
                activation=self.activation,
            )
        else:
            worker = ViT2D_CNN_Worker(
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
                final_architecture=self.final_architecture,
                block_channels=self.block_channels,
                kernel_size=self.kernel_size,
                n_ffn_layers_cnn=self.n_ffn_layers_cnn,
                activation=self.activation,
            )

        return jax.vmap(worker, in_axes=0)(x).squeeze()
