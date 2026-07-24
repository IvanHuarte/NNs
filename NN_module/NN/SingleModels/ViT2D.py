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
import jax.numpy as jnp
import jax.typing as jt

from ..toolbox import MultiLayerPerceptron, Translations2D

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
            .reshape(x.shape[0], *tok_lat_size, token_dim)
        )

    return nruter


class AffinityPosWeight(nn.Module):
    "Flax module that multiplies by a circulant matrix."

    token_lattice_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        weight_row = self.param(
            "alpha_delta",
            nn.initializers.truncated_normal(stddev=jnp.sqrt(1.0 / x.shape[-2])),
            (x.shape[-3], x.shape[-2]),
            DTYPE,
        )
        #
        weight = Translations2D(
            x=weight_row[None, ..., None], save_memory=False
        ).squeeze((-1, 0))

        # print(f"weight: {weight.shape}")
        # print(f"x: {x.shape}")

        y = jnp.einsum("ijkl,Bklc->Bijc", weight, x)
        # print(y.shape)
        return y


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
            * self.n_ffn_layers,
            # activation_function=nn.gelu,
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
