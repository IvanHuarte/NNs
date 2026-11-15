"""C4v wrapper: psi(g x) = chi(g) psi(x) for the 8 elements and the 4 one-dimensional irreps."""

import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN import REGISTRY_SYMM, symm_submodule_dict  # noqa: E402
from Hydra.NN.NN.Symm.C4v import C4V_IRREPS, C4v, c4v_characters, c4v_images  # noqa: E402

key = jax.random.PRNGKey(0)
x0 = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(3, 6, 6, 1))


class ToyK0(nn.Module):
    """Translation invariant (k=0) but not point-group invariant: periodic conv + site sum."""

    @nn.compact
    def __call__(self, x):
        y = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        y = jnp.tanh(nn.Conv(4, (3, 3), padding="VALID", param_dtype=jnp.float64)(y))
        return nn.Dense(1, param_dtype=jnp.float64)(y.sum(axis=(1, 2)))


@pytest.mark.parametrize("irrep", sorted(C4V_IRREPS))
def test_transforms_as_irrep(irrep):
    model = C4v(C4vWrap=ToyK0(), irrep=irrep)
    params = model.init(key, x0)
    psi = jnp.exp(model.apply(params, x0)).reshape(-1)
    chi = c4v_characters(irrep)
    for g, xg in enumerate(c4v_images(x0)):
        np.testing.assert_allclose(jnp.exp(model.apply(params, xg)).reshape(-1), chi[g] * psi, rtol=1e-9, atol=1e-12)


def test_not_trivial():
    """The wrapped toy is not C4v invariant, so the projection changes it."""
    toy = ToyK0()
    p = toy.init(key, x0)
    out = toy.apply(p, c4v_images(x0).reshape(-1, 6, 6, 1)).reshape(8, -1)
    assert not np.allclose(out[0], out[1])


def test_registered():
    assert REGISTRY_SYMM["C4v"] is C4v
    assert symm_submodule_dict["C4v"] == "C4vWrap"
