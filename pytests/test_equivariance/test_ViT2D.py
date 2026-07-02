import os
import sys
from time import time

import jax
import jax.numpy as jnp

from pytests.test_equivariance._trasl_equiv_check import (
    equivariance_traslation_all_test,
    equivariance_traslation_test,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.SingleModels.ViT2D import (
    AffinityPosWeight,
    MultiHeadPositionalAttention,
    PositionalHead,
    ViT2D,
    ViT2DBlock,
)
from NN_module.NN.SingleModels.VViT import EncoderBlock

equivariance_test = equivariance_traslation_test
equivariance_test = equivariance_traslation_all_test

key = jax.random.PRNGKey(int(time()))
lattice_size = (4, 1)
token_size = (1, 1)
token_lattice_size = (
    lattice_size[0] // token_size[0],
    lattice_size[1] // token_size[1],
)

embedding_d = 12
n_heads = 4
head_size = embedding_d // n_heads
n_blocks = 1
n_ffn_layers = 2
final_architecture = None

atol = 1e-10
verbosity = 1


def test_EncoderBlock():
    print("\nTesting EncoderBlock...\n")

    model = EncoderBlock(
        d_model=embedding_d,
        n_heads=n_heads,
        n_patches=lattice_size[0] * lattice_size[1],
        transl_invariant=True,
    )
    x0_shape = (1, token_lattice_size[0] * token_lattice_size[1], embedding_d)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_AffPosWeight():
    print("\nTesting AffinityPosWeight...\n")

    model = AffinityPosWeight(
        token_lattice_size=token_lattice_size,
    )
    x0_shape = (1, token_lattice_size[0] * token_lattice_size[1], head_size)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_PositionalHead():
    print("\nTesting PositionalHead...\n")
    model = PositionalHead(
        token_lattice_size=token_lattice_size,
        head_size=head_size,
    )
    x0_shape = (1, token_lattice_size[0] * token_lattice_size[1], head_size)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_MHPA():
    print("\nTesting MultiHeadPositionalAttention...\n")
    model = MultiHeadPositionalAttention(
        token_lattice_size=token_lattice_size,
        n_heads=n_heads,
        head_size=head_size,
    )
    x0_shape = (1, lattice_size[0] * lattice_size[1], embedding_d)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_ViT2DBlock():
    print("\nTesting ViT2DBlock...\n")

    model = ViT2DBlock(
        token_lattice_size=token_lattice_size,
        n_heads=n_heads,
        n_ffn_layers=n_ffn_layers,
    )
    x0_shape = (1, lattice_size[0] * lattice_size[1], embedding_d)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)


def test_ViT2D():

    print("\nTesting ViT2D...\n")
    model = ViT2D(
        lattice_size=lattice_size,
        token_size=token_size,
        embedding_d=embedding_d,
        n_heads=n_heads,
        n_blocks=n_blocks,
        n_ffn_layers=n_ffn_layers,
        final_architecture=final_architecture,
    )
    x0_shape = (1, lattice_size[0] * lattice_size[1], 1)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, lattice_size, params, model, atol=atol, v=verbosity)
