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
import netket as nk

from NN_module.NN_utils import traslations_2D
from ..toolbox import (
    MultiLayerPerceptron,
    two_heads,
    two_heads_phasors,
    glu_phasor,
)


REAL_DTYPE = jnp.asarray(1.0).dtype


class AffinityPosWeight(nn.Module):
    "Flax module that multiplies by a circulant matrix."

    token_lattice_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        # print(x.shape)

        # Traslation 2D
        if self.token_lattice_size is not None:
            weight_row = self.param(
                "alpha_delta",
                nn.initializers.truncated_normal(stddev=jnp.sqrt(1.0 / x.shape[-2])),
                (x.shape[-2],),
                REAL_DTYPE,
            )
            weight = traslations_2D(
                x=weight_row, size=self.token_lattice_size, memory=False
            )
        else:
            weight = self.param(
                "alpha_delta_nosymm",
                nn.initializers.truncated_normal(stddev=jnp.sqrt(1.0 / x.shape[-2])),
                (x.shape[-2], x.shape[-2]),
                REAL_DTYPE,
            )
            # weight = jnp.tile(weight_row, (x.shape[-2], 1))

        return weight @ x


class PositionalHead(nn.Module):
    """Flax module that implements to a single head of linearized attention.

    Args:
        head_size: The dimension of each of the heads.
    """

    token_lattice_size: Tuple[int, int]
    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        value = nn.Dense(self.head_size, use_bias=False, param_dtype=REAL_DTYPE)
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
        heads = [
            PositionalHead(self.token_lattice_size, self.head_size)
            for _ in range(self.n_heads)
        ]
        return jnp.concatenate([h(x) for h in heads], axis=-1)


class CoreBlock(nn.Module):
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
        x += sa(nn.LayerNorm(
            dtype=REAL_DTYPE,
            param_dtype=REAL_DTYPE
        )(x))
        # print(f"After attention: {x.shape}")
        ffn = MultiLayerPerceptron(
            [
                embedding_d,
            ]
            * self.n_ffn_layers
        )
        # print(f"After MLP: {ffn(x).shape}\n\n")
        # No LayerNorm here because it is already included in the perceptron.
        return nk.nn.log_cosh(ffn(x) + x)


class ViT2DWorker(nn.Module):
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

    token_lattice_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Tuple | None = None
    two_heads: bool = False
    phasors: bool = False

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]

        embedding = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)

        x = embedding(x)
        # print(f"x_embedd: {x.shape}")

        blocks = [
            CoreBlock(self.token_lattice_size, self.n_heads, self.n_ffn_layers)
            for _ in range(self.n_blocks)
        ]
        # print(f"Entering blocks: {len(blocks)} blocks with {self.n_heads} heads each. Number of FFN layers: {self.n_ffn_layers}")

        for cb in blocks:
            x = cb(x)
            # print(f"After CoreBlock: {x.shape}")

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
                x = nn.Dense(
                    1,
                    param_dtype=REAL_DTYPE
                )(MultiLayerPerceptron(self.final_architecture)(x))
                # print(f"Final: {x.shape}")

                return x


class TokenizeViT2D(nn.Module):
    """Flax module wrapping `ViT2DWorker` and enforcing Z2 and 2D-traslational invariance.

    This is achieved by averaging the result of `ViT2DWorker` over all
    possible cyclic permutations.

    See the documentation of `ViT2DWorker` for information about the
    parameters.
    """

    lattice_size: Tuple[int, int]
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Tuple | None = None
    two_heads: bool = False
    phasors: bool = False

    symm_2D: bool = False

    @nn.compact
    def __call__(self, x):
        # print(f"Tokenize:")
        # print(f"x: {x.shape}")

        B = x.shape[0]

        token_dim = self.token_size[0] * self.token_size[1]
        token_lattice_size = (
            self.lattice_size[0] // self.token_size[0],
            self.lattice_size[1] // self.token_size[1],
        )

        # Tokenizacion 2D
        x = (
            x.reshape(
                (
                    B,
                    token_lattice_size[1],
                    self.token_size[1],
                    token_lattice_size[0],
                    self.token_size[0],
                ),
                order="C",
            )
            .transpose((0, 1, 3, 2, 4))
            .reshape(B, -1, token_dim)
        )
        # print(f"x_tokenized: {x.shape}")

        token_size = self.token_size if self.symm_2D else None

        worker = ViT2DWorker(
            token_lattice_size=token_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
            two_heads=self.two_heads,
            phasors=self.phasors,
        )
        # print(f"x reshape: {x.shape}")

        return worker(x)


class ViT2D_2D(nn.Module):
    """Flax module wrapping `ViT2DWorker` and enforcing 2D-translation invariance.

    This is achieved by averaging the result of `ViT2DWorker` over all
    possible 2D translation permutations.

    See the documentation of `ViT2DWorker` for information about the
    parameters.
    """

    lattice_size: Tuple[int, int]
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Tuple | None = None
    two_heads: bool = False
    phasors: bool = False

    @nn.compact
    def __call__(self, x):
        # print(f"Input shape: {x.shape}")

        worker = TokenizeViT2D(
            token_size=self.token_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
            two_heads=self.two_heads,
            phasors=self.phasors,
            symm_2D=True,
        )
        # print(self.token_size, type(self.token_size))

        # 2D traslation
        traslational_x = traslations_2D(  # (N_tr, B, N)
            x, size=self.lattice_size, memory=False
        )

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class ViT2D_Z2(nn.Module):
    """Flax module wrapping `ViT2DWorker` and enforcing Z2 and 2D-traslational invariance.

    This is achieved by averaging the result of `ViT2DWorker` over all
    possible cyclic permutations.

    See the documentation of `ViT2DWorker` for information about the
    parameters.
    """

    lattice_size: Tuple[int, int]
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Tuple | None = None
    two_heads: bool = False
    phasors: bool = False

    trivial_Z2: bool = True
    symm_2D: bool = False

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = ViT2D_2D(
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
                final_architecture=self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
            )

        else:
            worker = TokenizeViT2D(
                token_size=self.token_size,
                embedding_d=self.embedding_d,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                n_ffn_layers=self.n_ffn_layers,
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


class ViT2D(nn.Module):
    "Batched version of ViT2D, accepting several spin configurations at once."

    lattice_size: Tuple[int, int]
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int

    final_architecture: Tuple | None = None
    two_heads: bool = False
    phasors: bool = False

    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):
        if self.symm_Z2:
            worker = ViT2D_Z2(
                self.lattice_size,
                self.token_size,
                self.embedding_d,
                self.n_heads,
                self.n_blocks,
                self.n_ffn_layers,
                self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
            )
        elif self.symm_2D:
            worker = ViT2D_2D(
                self.lattice_size,
                self.token_size,
                self.embedding_d,
                self.n_heads,
                self.n_blocks,
                self.n_ffn_layers,
                self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
            )
        else:
            worker = TokenizeViT2D(
                self.lattice_size,
                self.token_size,
                self.embedding_d,
                self.n_heads,
                self.n_blocks,
                self.n_ffn_layers,
                self.final_architecture,
                two_heads=self.two_heads,
                phasors=self.phasors,
            )

        return worker(x)
