"""Real wrapper (arXiv:2611.18840): psi + psi*, real log-derivatives, commutes with C4v."""

import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN import REGISTRY_SYMM, symm_submodule_dict  # noqa: E402
from Hydra.NN.NN.Symm.C4v import C4v  # noqa: E402
from Hydra.NN.NN.Symm.Real import Real, real_project_logpsi  # noqa: E402

key = jax.random.PRNGKey(0)
x0 = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(5, 6, 6, 1))


class ToyComplex(nn.Module):
    """k=0, complex log psi with phases spread over (-pi, pi]."""

    @nn.compact
    def __call__(self, x):
        y = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        y = jnp.tanh(nn.Conv(4, (3, 3), padding="VALID", param_dtype=jnp.float64)(y)).sum(axis=(1, 2))
        a, b = nn.Dense(1, param_dtype=jnp.float64)(y), nn.Dense(1, param_dtype=jnp.float64)(y)
        return a + 3j * b


def test_is_twice_the_real_part():
    lp = jax.random.normal(key, (200,)) * 2 + 1j * jax.random.uniform(key, (200,), minval=-9, maxval=9)
    out = real_project_logpsi(lp)
    np.testing.assert_allclose(jnp.exp(out), jnp.exp(lp).real, rtol=1e-10, atol=1e-12)
    # the formula of the paper's code, log psi + log1p(exp(-2i Im log psi)) = log(2 Re psi)
    paper = lp + jnp.log1p(jnp.exp(-2.0j * lp.imag))
    np.testing.assert_allclose(jnp.exp(paper), 2 * jnp.exp(out), rtol=1e-9, atol=1e-12)
    assert set(np.unique(np.round(np.asarray(out.imag), 12))) <= {0.0, round(np.pi, 12)}


def test_log_derivatives_are_real():
    model = Real(RealWrap=ToyComplex())
    p = model.init(key, x0)
    jac = jax.jacrev(lambda p: model.apply(p, x0).reshape(-1).imag)(p)
    assert all(np.allclose(j, 0) for j in jax.tree.leaves(jac))
    jac_re = jax.jacrev(lambda p: model.apply(p, x0).reshape(-1).real)(p)
    assert any(not np.allclose(j, 0) for j in jax.tree.leaves(jac_re))


@pytest.mark.parametrize("irrep", ["A1", "B2"])
def test_commutes_with_c4v(irrep):
    """Real characters: Re sum_g chi psi(gx) = sum_g chi Re psi(gx), either nesting, same parameters."""
    outer = Real(RealWrap=C4v(C4vWrap=ToyComplex(), irrep=irrep))
    inner = C4v(C4vWrap=Real(RealWrap=ToyComplex()), irrep=irrep)
    p = outer.init(key, x0)
    leaf = p["params"]["RealWrap"]["C4vWrap"]
    q = {"params": {"C4vWrap": {"RealWrap": leaf}}}
    np.testing.assert_allclose(jnp.exp(outer.apply(p, x0)), jnp.exp(inner.apply(q, x0)), rtol=1e-8, atol=1e-12)


def test_registered():
    assert REGISTRY_SYMM["Real"] is Real
    assert symm_submodule_dict["Real"] == "RealWrap"
