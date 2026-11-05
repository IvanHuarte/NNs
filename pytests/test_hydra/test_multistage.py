"""
Multi-stage Hydra: architecture evolution + weight transplantation.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_hydra -q
"""

import copy
import os
import sys

import jax
import jax.numpy as jnp
import netket as nk
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.Hydra import Hydra
from NN_module.NN.utils import flatten_with_paths

LATTICE = (4, 4)
N = LATTICE[0] * LATTICE[1]

STORAGE = {
    "ViT2D": {
        "token_size": [1, 2],
        "embedding_d": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "n_ffn_layers": 1,
        "final_architecture": None,
    },
    "ComplexHead": {"d_model": 8, "only_phase": True},
    "OutputHead": {"d_model": 8, "only_phase": True},
}

SYMM_WRAPPER = {
    "Traslation": {
        "on": False,
        "irrep": [0, 0],
        "use_anchor": False,
        "save_memory": False,
    },
    "Z2": {"on": False, "trivial_Z2": True, "symm_modulus": False},
}


def make_config(evolution):
    return {
        "print_arch": False,
        "load_model": {"parameters": False, "sampler": False},
        "storage": copy.deepcopy(STORAGE),
        "symm_wrapper": copy.deepcopy(SYMM_WRAPPER),
        "selection": "Test",
        "Test": evolution,
    }


def build(config):
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    return Hydra(config, lattice_size=LATTICE, hilbert=hi)


def init_params(hydra, seed):
    return hydra.model.init(jax.random.PRNGKey(seed), jnp.ones((1, N)))["params"]


def log_psi(hydra, params, x):
    return hydra.model.apply({"params": params}, x)


def configs(seed=123, B=32):
    return jax.random.choice(jax.random.PRNGKey(seed), jnp.array([-1.0, 1.0]), (B, N))


def evolve(hydra, old_params, seed=7):
    hydra.arch_evol(old_params)
    new_params = init_params(hydra, seed)
    return hydra.weight_transplantation(new_params)


def leaves_equal(a, b):
    fa, fb = flatten_with_paths(a), flatten_with_paths(b)
    return len(fa) == len(fb) and all(
        pa == pb and jnp.array_equal(xa, xb) for (pa, xa), (pb, xb) in zip(fa, fb)
    )


VIT_HEAD = {"Sequential": {"ViT2D": {}, "ComplexHead": {}}}
VIT_OUT = {"Sequential": {"ViT2D": {}, "OutputHead": {}}}


def test_stage_reference_builds_without_double_wrapper():
    """{"stage0": {}} reuses stage0 network wrapped only once."""
    config = make_config(
        {
            "stage0": {"template": VIT_HEAD, "save_params": True},
            "stage1": {"template": {"stage0": {}}},
        }
    )
    hydra = build(config)
    p0 = init_params(hydra, 0)
    p1 = evolve(hydra, p0)

    stage1 = hydra.storage["stage1"]
    assert stage1["module"] == "NNWrapper"
    assert stage1["setup"]["module"] == "Sequential"
    # Same architecture -> identical tree structure and codes
    assert hydra.get_code2path(p0) == hydra.get_code2path(p1)


def test_default_load_transplants_everything():
    """No 'load' -> all compatible weights copied: same log psi."""
    config = make_config(
        {
            "stage0": {"template": VIT_HEAD, "save_params": False},
            "stage1": {"template": {"stage0": {}}},
        }
    )
    hydra = build(config)
    p0 = init_params(hydra, 0)
    x = configs()
    lp0 = log_psi(hydra, p0, x)

    p1 = evolve(hydra, p0)
    assert leaves_equal(p0, p1)
    assert jnp.allclose(lp0, log_psi(hydra, p1, x))


def test_explicit_empty_load_keeps_random_init():
    config = make_config(
        {
            "stage0": {"template": VIT_HEAD, "save_params": True},
            "stage1": {"template": {"stage0": {}}, "load": []},
        }
    )
    hydra = build(config)
    p0 = init_params(hydra, 0)
    p1 = evolve(hydra, p0, seed=7)
    assert leaves_equal(p1, init_params(hydra, 7))
    assert not leaves_equal(p0, p1)


def test_change_attr_is_applied_and_config_untouched():
    evolution = {
        "stage0": {"template": VIT_HEAD, "save_params": True},
        "stage1": {
            "template": {"stage0": {}},
            "change_attr": {"ComplexHead": {"only_phase": False}},
        },
    }
    config = make_config(evolution)
    config_before = copy.deepcopy(config)

    hydra = build(config)
    p0 = init_params(hydra, 0)
    evolve(hydra, p0)

    head0 = hydra.storage["stage0"]["setup"]["setup"]["Seq_1"]["setup"]
    head1 = hydra.storage["stage1"]["setup"]["setup"]["Seq_1"]["setup"]
    assert head0["only_phase"] is True
    assert head1["only_phase"] is False
    assert hydra.model.NN.Seq[1].only_phase is False
    # Neither the user config nor the storage templates were modified
    assert config == config_before
    assert hydra.storage["ComplexHead"]["only_phase"] is True


def test_symm_wrapper_turned_on_in_later_stage():
    """Like CvT2OutputHead2Phase: Z2 on + explicit load codes."""
    config = make_config(
        {
            "stage0": {"template": VIT_OUT, "save_params": True},
            "stage1": {
                "template": {"stage0": {}},
                "symm_wrapper": {"Z2": {"on": True}},
                "load": [["stage0", "0", "0"]],
            },
        }
    )
    hydra = build(config)
    p0 = init_params(hydra, 0)
    p1 = evolve(hydra, p0)

    assert hydra.symm_wrapper["Z2"]["on"] is True
    assert hydra.storage["stage1"]["setup"]["module"] == "Z2"
    # Z2 adds no parameters: every array is transplanted to the same place
    leaves0 = flatten_with_paths(p0)
    leaves1 = flatten_with_paths(p1)
    assert len(leaves0) == len(leaves1)
    strip = lambda p: tuple(k for k in p if k != "Z2Wrap")
    d1 = {strip(p): x for p, x in leaves1}
    assert all(jnp.array_equal(x, d1[p]) for p, x in leaves0)


def test_unknown_symm_option_raises():
    config = make_config(
        {
            "stage0": {"template": VIT_HEAD},
            "stage1": {"template": {"stage0": {}}, "symm_wrapper": {"Z2": {"bad": 1}}},
        }
    )
    hydra = build(config)
    with pytest.raises(AssertionError):
        hydra.arch_evol(init_params(hydra, 0))


def test_mcstate_stage_change_like_vmc_simulation():
    """Reproduces the stage-change block of VMC_simulation.py."""
    config = make_config(
        {
            "stage0": {"template": VIT_HEAD, "save_params": True},
            "stage1": {
                "template": {"stage0": {}},
                "change_attr": {"ComplexHead": {"only_phase": False}},
            },
        }
    )
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    hydra = Hydra(config, lattice_size=LATTICE, hilbert=hi)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=8)
    vstate = nk.vqs.MCState(sampler, hydra.model, n_samples=64)
    p0 = vstate.parameters

    hydra.arch_evol(vstate.parameters)
    vstate = nk.vqs.MCState(sampler, hydra.model, n_samples=64)
    vstate.parameters = hydra.weight_transplantation(vstate.parameters)
    code2path = hydra.get_code2path(vstate.parameters)

    assert code2path == hydra.get_code2path(p0)
    # The ViT body is fully transplanted
    assert leaves_equal(p0["NN"]["Seq_0"], vstate.parameters["NN"]["Seq_0"])
    vstate.sample()
