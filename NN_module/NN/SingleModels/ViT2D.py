# Copyright 2024 the authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import numpy.typing as npt

from ..toolbox import MultiLayerPerceptron, traslations_2D

DTYPE = jnp.float64


def Tokenize(token_size, tok_lat_size, token_dim):
    def nruter(x):
        return (
            x.reshape(
                (
                    x.shape[0],
                    tok_lat_size[1],
                    token_size[1],
                    tok_lat_size[0],
                    token_size[0],
                ),
                order="C",
            )
            .transpose((0, 1, 3, 2, 4))
            .reshape(x.shape[0], -1, token_dim)
        )

    return nruter


def make_circulant(tile: npt.ArrayLike, roll_axis: int = -1) -> npt.ArrayLike:
    """Add an axis to an array making it circulant.

    Args:
        tile: the building block of the circulant array.
        roll_axis: axis over which to roll the tile each time it is repeated.

    Returns:
        A new array with all the offset versions of the tile.
    """
    tile = jnp.asarray(tile)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1, axis=roll_axis)
        return (new_carry, new_carry)

    nruter = jax.lax.scan(scan_arg, tile, length=tile.shape[roll_axis])[1][::-1, ...]

    return nruter


def create_2D_circulant_from_motif(motif: npt.ArrayLike) -> npt.ArrayLike:
    """Create a circulant attention array for a 2D system from a 2D motif.

    Args:
        motif: the repeating motif.

    Returns:
        The attention (a 4D array).
    """
    with_three_axes = make_circulant(motif, 1)
    with_four_axes = make_circulant(with_three_axes, 1)

    return with_four_axes


class AffinityPosWeight2D(nn.Module):
    "Flax module that implements a circular positional attention in 2D."

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        weight_motif = self.param(
            "alpha_delta",
            nn.initializers.glorot_normal(),
            (x.shape[-3], x.shape[-2]),
            DTYPE,
        )
        print(x.shape)
        # weight = create_2D_circulant_from_motif(weight_motif)

        # MY WAY
        weight_motif = weight_motif[None, ...].reshape(1, x.shape[-3] * x.shape[-2], -1)
        weight = (
            traslations_2D(x=weight_motif, size=(4, 4), memory=False)
            .reshape(1, x.shape[-3], x.shape[-2], x.shape[-3], x.shape[-2])
            .squeeze()
        )
        print(weight.shape)

        return jnp.einsum("ijkl,klm->ijm", weight, x)


class PositionalHead2D(nn.Module):
    """Flax module that implements a single head of linearized attention in 2D.

    Args:
        head_size: The dimension of the head.
    """

    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        value = nn.Dense(self.head_size, use_bias=False, param_dtype=DTYPE)
        aff = AffinityPosWeight2D()

        return aff(value(x))


class TestModule(nn.Module):

    lattice_size: Tuple[int, int]
    token_lattice_size: Tuple[int, int]
    head_size: int
    n_heads: int
    embedding_d: int
    token_size: Tuple

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        print(x.shape, self.token_lattice_size)
        token_dim = self.token_size[0] * self.token_size[1]
        token_lattice_size = (
            self.lattice_size[0] // self.token_size[0],
            self.lattice_size[1] // self.token_size[1],
        )

        x = Tokenize(self.token_size, token_lattice_size, token_dim)(x)
        x = nn.Dense(self.embedding_d, param_dtype=DTYPE)(x)
        # Equivariante hasta aqui

        head_size = self.embedding_d // self.n_heads

        # x = MultiHeadPositionalAttention(
        #     self.token_lattice_size, self.n_heads, head_size
        # )(x)

        return x


# class AffinityPosWeight(nn.Module):
#     "Flax module that multiplies by a circulant matrix."

#     token_lattice_size: Tuple[int, int]

#     @nn.compact
#     def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

#         # Traslation 2D
#         weight_row = self.param(
#             "alpha_delta",
#             nn.initializers.truncated_normal(stddev=jnp.sqrt(1.0 / x.shape[-2])),
#             (x.shape[-2],),
#             DTYPE,
#         )
#         weight = traslations_2D(
#             x=weight_row, size=self.token_lattice_size, memory=False
#         )

#         print(f"weight: {weight.shape}")
#         print(f"x: {x.shape}")
#         # print((weight @ x).shape)
#         y = jnp.einsum("mn,inc->imc", weight, x)
#         print(y.shape)
#         return y.reshape(y.shape[0], *self.token_lattice_size, y.shape[-1])


# return weight @ x.
#
#
class AffinityPosWeight(nn.Module):
    "Flax module that multiplies by a circulant matrix."

    #
    token_lattice_size: Tuple[int, int]

    #
    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        x_in_shape = x.shape
        x = x.reshape(x.shape[0], *self.token_lattice_size, x.shape[-1])
        #
        # Traslation 2D
        weight_row = self.param(
            "alpha_delta",
            nn.initializers.truncated_normal(stddev=jnp.sqrt(1.0 / x.shape[-2])),
            (x.shape[-3], x.shape[-2]),
            DTYPE,
        )
        #
        weight = weight_row[None, ...].reshape(1, x.shape[-3] * x.shape[-2], -1)
        weight = (
            traslations_2D(x=weight, size=(4, 4), memory=False)
            .reshape(1, x.shape[-3], x.shape[-2], x.shape[-3], x.shape[-2])
            .squeeze()
        )
        #
        print(f"weight: {weight.shape}")
        print(f"x: {x.shape}")
        # print((weight @ x).shape)
        y = jnp.einsum("ijkl,Bklc->Bijc", weight, x)
        print(y.shape)
        return y.reshape(*x_in_shape)


class PositionalHead(nn.Module):
    """Flax module that implements to a single head of linearized attention.

    Args:
        head_size: The dimension of each of the heads.
    """

    token_lattice_size: Tuple[int, int]
    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"PositionalHead: {x.shape}")
        value = nn.Dense(self.head_size, use_bias=False, param_dtype=DTYPE)
        aff = AffinityPosWeight(self.token_lattice_size)

        return aff(value(x))


class MultiHeadPositionalAttention(nn.Module):
    """Flax module implementing a multi-head block.

    The input is split into the chosen number of heads and the result is
    concatenated.

    Args:
        n_heads: The number of heads present in the multi-head attention block.
        head_size: The dimension of each one of the heads.
    """

    token_lattice_size: Tuple[int, int]
    n_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(f"x.shape: {x.shape}")
        heads = [
            PositionalHead(self.token_lattice_size, self.head_size)
            for _ in range(self.n_heads)
        ]
        return jnp.concatenate([h(x) for h in heads], axis=-1)


class ViT2DBlock(nn.Module):
    """Flax module implementing the core block of the ViT.

    This comprises the embedding starting from the spin representation, the
    application of the multi-head linearized attention block and the multi-layer
    perceptron, and a final pass through a log-cosh function to achieve higher
    accuracy in highly ordered phases. All the operations here are applied to
    all tokens indistincly.

    Args:
        n_heads: The number of heads present in the multi-head attention block.
        n_ffn_layers: The number of layers present in the multi-layer perceptron.
    """

    token_lattice_size: Tuple[int, int]
    n_heads: int
    n_ffn_layers: int

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        embedding_d = x.shape[-1]
        # print(f"Input shape: {x.shape}")
        if embedding_d % self.n_heads != 0:
            raise ValueError("The number of heads must divide the embedding dimensions")
        head_size = embedding_d // self.n_heads
        sa = MultiHeadPositionalAttention(
            self.token_lattice_size, self.n_heads, head_size
        )
        sax = sa(nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x))
        x += sax
        # print(f"After attention: {x.shape}")
        ffn = MultiLayerPerceptron(
            [
                embedding_d,
            ]
            * self.n_ffn_layers
        )
        # print(f"After MLP: {ffn(x).shape}\n\n")
        # No LayerNorm here because it is already included in the perceptron.
        return ffn(x) + x


class ViT2D(nn.Module):
    """Flax module that implements a real-valued ViT architecture.

    Two of these modules can be combined to create a complex output. The input
    is expected to be passed configuration by configuration. Note that the
    tokenization is not including.

    Args:
        embedding_d: The dimension of the linear transformation that maps the
            binary spin representation into the feature space onto which the ViT
            will act.
        n_heads: The number of heads present in the Multi-Head attention block.
        n_blocks: The number of core blocks present in the model.
        n_ffn_layers: The number of layers present in the multi-layer perceptron
            contained in the CoreBlock.
        final_architecture: A sequence of integer widths for the layers in the
            post-processing MLP.
    """

    token_size: Tuple[int, int]

    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Tuple | None = None

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]

        token_dim = self.token_size[0] * self.token_size[1]
        token_lattice_size = (
            x.shape[1] // self.token_size[0],
            x.shape[2] // self.token_size[1],
        )

        x = Tokenize(self.token_size, token_lattice_size, token_dim)(x)
        embedding = nn.Dense(self.embedding_d, param_dtype=DTYPE)

        x = embedding(x)
        # print(f"x_embedd: {x.shape}")

        blocks = [
            ViT2DBlock(token_lattice_size, self.n_heads, self.n_ffn_layers)
            for _ in range(self.n_blocks)
        ]
        # print(f"Entering blocks: {len(blocks)} blocks with {self.n_heads} heads each. Number of FFN layers: {self.n_ffn_layers}")

        for cb in blocks:
            x = cb(x)
            # print(f"After CoreBlock: {x.shape}")

        # Works with termination module by default
        if self.final_architecture is None:
            return x

        else:
            x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)  # Mean pooling over spins
            for hi in self.final_architecture:
                x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
            return x
