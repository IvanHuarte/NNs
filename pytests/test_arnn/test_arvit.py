"""
ARViT + ARNNWrapper must work with an unmodified NetKet (no patched
netket/models/autoreg.py).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_arnn -q
"""

import os
import sys

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN.Hydra import Hydra

LATTICE = (2, 3)
N = LATTICE[0] * LATTICE[1]
HI = nk.hilbert.Spin(s=1 / 2, N=N)


def make_hydra():
    config = {
        "print_arch": False,
        "load_model": {"parameters": False, "sampler": False},
        "storage": {
            "ARViT": {
                "embedding_d": 8,
                "n_heads": 2,
                "n_blocks": 1,
                "n_ffn_layers": 1,
                "machine_pow": 2,
            }
        },
        "symm_wrapper": {},
        "selection": "Test",
        "Test": {"stage0": {"template": {"ARViT": {}}}},
    }
    return Hydra(config, lattice_size=LATTICE, hilbert=HI)


def nonzero_params(model):
    """ARViT's final layer is zero-initialized: perturb so psi is non-trivial."""
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, N)))
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    leaves = [x + 0.3 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(tree, leaves)


def test_wrapper_is_autoregressive():
    hydra = make_hydra()
    assert hydra.setup["module"] == "ARNNWrapper"
    assert isinstance(hydra.model, nk.models.AbstractARNN)


def test_input_shapes_agree():
    """(N,), (B, N) and (B, Lx, Ly, 1) give the same log psi."""
    model = make_hydra().model
    variables = nonzero_params(model)
    x = HI.all_states()[:5]

    lp_flat = model.apply(variables, x)
    lp_2d = model.apply(variables, x.reshape(-1, *LATTICE, 1))
    lp_one = model.apply(variables, x[0])

    assert lp_flat.shape == (5,)
    np.testing.assert_allclose(lp_flat, lp_2d)
    np.testing.assert_allclose(lp_flat[0], lp_one[0])


def test_normalized():
    """Autoregressive models are normalized: sum |psi|^2 over all states = 1."""
    model = make_hydra().model
    variables = nonzero_params(model)
    lp = model.apply(variables, HI.all_states())
    norm = jnp.sum(jnp.exp(2 * lp.real))
    np.testing.assert_allclose(norm, 1.0, rtol=1e-6)


def test_log_psi_matches_conditionals():
    """__call__ equals the sum of the selected conditionals."""
    model = make_hydra().model
    variables = nonzero_params(model)
    x = HI.all_states()
    cond = model.apply(variables, x, method=model.conditionals_log_psi)
    idx = HI.states_to_local_indices(x)
    expected = jnp.take_along_axis(cond, idx[..., None], axis=-1)[..., 0].sum(-1)
    np.testing.assert_allclose(model.apply(variables, x), expected)


def test_direct_sampler_and_mcstate(capfd):
    """ARDirectSampler draws exact samples; no debug output is printed."""
    model = make_hydra().model
    sampler = nk.sampler.ARDirectSampler(HI)
    vstate = nk.vqs.MCState(sampler, model, n_samples=64, seed=0, sampler_seed=1)
    vstate.parameters = nonzero_params(model)["params"]
    capfd.readouterr()

    samples = vstate.sample()
    assert samples.shape[-1] == N
    assert set(np.unique(np.asarray(samples))) <= {-1.0, 1.0}

    H = nk.operator.Ising(HI, nk.graph.Chain(N), h=1.0)
    E = vstate.expect(H)
    assert np.isfinite(E.mean)

    out = capfd.readouterr().out
    assert "conditioals" not in out and "inputs:" not in out


def test_independent_of_netket_autoreg_implementation():
    """The methods NetKet's AbstractARNN provides (and that were patched by
    hand in site-packages) are redefined in our classes."""
    from Hydra.NN.ARNN import ARNNWrapper
    from Hydra.NN.ARNN.SingleModules.ARViT import ARViT

    assert "__call__" in ARViT.__dict__
    assert "__call__" in ARNNWrapper.__dict__
    assert "conditional" in ARNNWrapper.__dict__
