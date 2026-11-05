"""
Sampler configuration (config0 "sampler") and Hamiltonian units (config1
"S_operators").

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_sampler -q
"""

import copy
import os
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from NN_module.sampler.custom_rules import MultipleRules
from NN_module.sampler.sampler import SamplerFactory, resolve_sampler_setup

N = 6
HI = nk.hilbert.Spin(s=1 / 2, N=N)

SETUP = {
    "selection": "metropolis",
    "metropolis": {
        "sampler": {"rules": ["LocalRule"], "prules": [1.0], "initial_state": None},
        "n_chains_per_rank": 8,
        "n_samples_per_chain": 4,
    },
    "nkARSampler": {},
    "n_chains_per_rank": 32,
    "n_samples_per_chain": 32,
    "chunk_sampler": None,
    "chunk_vstate": 1024,
}


def setup(**sampler_overrides):
    s = copy.deepcopy(SETUP)
    s["metropolis"]["sampler"].update(sampler_overrides)
    return s


# ------------------------------------------------- (a) selected block wins


def test_selected_block_overrides_general_values(capsys):
    resolved = resolve_sampler_setup(SETUP, verbose=True)
    assert resolved["n_chains_per_rank"] == 8
    assert resolved["n_samples_per_chain"] == 4
    assert resolved["chunk_vstate"] == 1024  # not in the block: general value
    assert "n_chains_per_rank = 8 (general value 32 ignored)" in capsys.readouterr().out


def test_other_blocks_are_ignored_and_resolution_is_idempotent():
    s = copy.deepcopy(SETUP)
    s["selection"] = "nkARSampler"
    resolved = resolve_sampler_setup(s)
    assert resolved["n_chains_per_rank"] == 32
    assert resolve_sampler_setup(resolve_sampler_setup(SETUP)) == resolve_sampler_setup(SETUP)
    assert SETUP["n_chains_per_rank"] == 32  # input not modified


def test_metropolis_sampler_uses_selected_block_chains():
    sampler = SamplerFactory(SETUP).get_sampler(HI)
    assert sampler.n_chains == 8


# ------------------------------------------------------ (b) initial_state


def test_single_rule_without_initial_state_is_the_rule_itself():
    sampler = SamplerFactory(setup()).get_sampler(HI)
    assert isinstance(sampler.rule, nk.sampler.rules.LocalRule)


@pytest.mark.parametrize("rules,prules", [(["LocalRule"], [1.0]),
                                          (["LocalRule", "InvertMagnetization"], [0.5, 0.5])])
def test_initial_state_is_applied(rules, prules):
    """M0: all chains start in the zero-magnetization sector."""
    sampler = SamplerFactory(setup(rules=rules, prules=prules, initial_state="M0")).get_sampler(HI)
    assert isinstance(sampler.rule, MultipleRules)
    assert sampler.rule.initial_state == "M0"

    model = nk.models.RBM(alpha=1, param_dtype=float)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, N)))
    state = sampler.init_state(model, params, seed=0)
    state = sampler.reset(model, params, state)
    assert jnp.all(state.σ.sum(axis=-1) == 0)


def test_unknown_initial_state_raises():
    with pytest.raises(ValueError, match="Unknown initial_state 'M1'"):
        SamplerFactory(setup(initial_state="M1")).get_sampler(HI)


# -------------------------------------------------------------- (c) prules


def test_prules_length_mismatch_gives_clear_error():
    s = setup(rules=["LocalRule", "InvertMagnetization"], prules=[1.0])
    with pytest.raises(ValueError, match="2 rules .* but 1 probabilities"):
        SamplerFactory(s).get_sampler(HI)


def test_prules_must_sum_to_one():
    s = setup(rules=["LocalRule", "InvertMagnetization"], prules=[0.5, 0.6])
    with pytest.raises(ValueError, match="must sum to 1"):
        SamplerFactory(s).get_sampler(HI)


def test_single_rule_ignores_prules():
    sampler = SamplerFactory(setup(prules=[0.5, 0.5])).get_sampler(HI)
    assert isinstance(sampler.rule, nk.sampler.rules.LocalRule)


# ------------------------------------------------------- (d) S_operators


VA_SRC = os.path.abspath(os.path.join(ROOT, "../ATMOS_VA/VA/src"))


@pytest.fixture
def Runner():
    if not os.path.isdir(os.path.join(VA_SRC, "VA_project")):
        pytest.skip("VA_project not found")
    sys.path.insert(0, VA_SRC)
    from VA_project.engine.runners import Runner

    return Runner


def chain_cm(J, h):
    """Heisenberg chain with uniform Z field, in the VA_project cm format."""
    Q = 4
    couplings = np.zeros((3, Q, Q))
    for i in range(Q - 1):
        couplings[:, i, i + 1] = J
    return SimpleNamespace(
        operators=[["Z"], ["XX", "YY", "ZZ"]],
        onsite_terms=np.full((1, Q), h),
        coupling_terms=couplings,
        lattice=SimpleNamespace(Q=Q),
    )


def energies(Runner, J, h, S_operators):
    H = Runner(chain_cm(J, h), S_operators=S_operators).build_hamiltonian()
    return np.linalg.eigvalsh(H.to_dense())


def test_couplings_scale_by_4_fields_by_2(Runner):
    """S = sigma/2: two-spin terms scale by 1/4, one-spin terms by 1/2.
    Hence the old 'E / 4' was wrong whenever there are fields."""
    np.testing.assert_allclose(energies(Runner, 1.0, 0.0, True), energies(Runner, 1.0, 0.0, False) / 4)
    np.testing.assert_allclose(energies(Runner, 0.0, 1.0, True), energies(Runner, 0.0, 1.0, False) / 2)
    E_S = energies(Runner, 1.0, 0.7, True)[0]
    E_P = energies(Runner, 1.0, 0.7, False)[0]
    assert not np.isclose(E_S, E_P / 4)
