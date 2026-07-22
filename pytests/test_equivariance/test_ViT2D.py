import os
import sys
from time import time

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.SingleModels.ViT2D import (
    AffinityPosWeight,
    MultiHeadPositionalAttention,
    PositionalHead,
    ViT2D,
    ViT2DBlock,
)
from NN_module.NN.SingleModels.FluxFunc import OutputHead
from NN_module.NN.Factories.Sequivariant import Sequivariant
from pytests.test_equivariance._trasl_equiv_check import (
    traslation_equivariant, traslation_equivariant_irrep
)

equivariance_test = traslation_equivariant
equivariance_test_irrep = traslation_equivariant_irrep

key = jax.random.PRNGKey(int(time()))
lattice_size = (4, 4)
token_size = (1, 1)
token_lattice_size = (
    lattice_size[0] // token_size[0],
    lattice_size[1] // token_size[1],
)

embedding_d = 60
n_heads = 10
head_size = embedding_d // n_heads
n_blocks = 1
n_ffn_layers = 2
final_architecture = None

irrep = (3, 2)
atol = 1e-10
verbosity = 1




def test_AffPosWeight():
    print("\nTesting AffinityPosWeight...\n")

    model = AffinityPosWeight(
        token_lattice_size=token_lattice_size,
    )
    x0_shape = (1, token_lattice_size[0], token_lattice_size[1], head_size)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, params, model, atol=atol, v=verbosity)


def test_PositionalHead():
    print("\nTesting PositionalHead...\n")
    model = PositionalHead(
        token_lattice_size=token_lattice_size,
        head_size=head_size,
    )
    x0_shape = (1, token_lattice_size[0], token_lattice_size[1], embedding_d)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, params, model, atol=atol, v=verbosity)


def test_MHPA():
    print("\nTesting MultiHeadPositionalAttention...\n")
    model = MultiHeadPositionalAttention(
        token_lattice_size=token_lattice_size,
        n_heads=n_heads,
        head_size=head_size,
    )
    x0_shape = (1, token_lattice_size[0], token_lattice_size[1], embedding_d)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, params, model, atol=atol, v=verbosity)


def test_ViT2DBlock():
    print("\nTesting ViT2DBlock...\n")

    model = ViT2DBlock(
        token_lattice_size=token_lattice_size,
        n_heads=n_heads,
        n_ffn_layers=n_ffn_layers,
    )
    x0_shape = (1, token_lattice_size[0], token_lattice_size[1], embedding_d)
    x0 = jax.random.uniform(key, shape=x0_shape, minval=-2.0, maxval=2.0)
    params = model.init(key, x0)
    return equivariance_test(x0, params, model, atol=atol, v=verbosity)


def test_ViT2D():

    print("\nTesting ViT2D...\n")
    model = ViT2D(
        token_size=token_size,
        embedding_d=embedding_d,
        n_heads=n_heads,
        n_blocks=n_blocks,
        n_ffn_layers=n_ffn_layers,
        final_architecture=final_architecture,
    )
    x0_shape = (1, lattice_size[0], lattice_size[1], 1)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)
    return equivariance_test(x0, params, model, atol=atol, v=verbosity)

def test_ViT2D_irreps():
    from itertools import product
    verbosity = 0
    irreps = product(range(lattice_size[0]), range(lattice_size[1]))
    print("\nTesting ViT2D...\n")
    for irrep in irreps:
        print(f"IRREP: {irrep}")
            
        vit = ViT2D(
            token_size=token_size,
            embedding_d=embedding_d,
            n_heads=n_heads,
            n_blocks=n_blocks,
            n_ffn_layers=n_ffn_layers,
            final_architecture=final_architecture,
        )
        final = OutputHead(
            d_model=embedding_d
        )
        model = Sequivariant(
            SeqV=(vit, final),
            irrep=irrep
        )
        x0_shape = (1, lattice_size[0], lattice_size[1], 1)
        x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
        params = model.init(key, x0)
        traslation_equivariant_irrep(x0, params, model, irrep, atol=atol, v=verbosity)
    return 
