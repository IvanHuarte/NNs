"""
Path codes of the network parameters and their use in the Schedule.

Codes index the generic network tree: "0", "1", ... are the first-level
submodules of the network itself, "01020" a deeper node, etc. Top wrappers
(NNWrapper 'NN', ARNNWrapper 'ARNN') and symmetry
wrappers ('Z2Wrap', 'TWrap') do not consume a digit.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_schedule -q
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
from NN_module.NN.utils import (
    flatten_with_paths,
    get_code2path_flatten,
    get_code2path_tree,
    print_architecture,
    upgrade_legacy_params,
)
from NN_module.schedule.schedule import Schedule

LATTICE = (4, 4)
N = 16

STORAGE = {
    "ViT2D": {
        "token_size": [1, 2],
        "embedding_d": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "n_ffn_layers": 1,
        "final_architecture": None,
    },
    "ComplexHead": {"d_model": 8, "only_phase": False},
}
SYMM = {
    "Traslation": {"on": False, "irrep": [0, 0], "use_anchor": False, "save_memory": False},
    "Z2": {"on": False, "trivial_Z2": True, "symm_modulus": False},
}
OPTIMIZER = {
    "optimizer": "sgd",
    "modifications": {
        "clip": None,
        "clip_by_block_rms": None,
        "scale": None,
        "scale_by_rms": None,
        "scale_by_learning_rate": None,
        "add_noise": None,
        "trace": None,
        "clip_by_global_norm": None,
    },
}


def build(z2=False):
    symm = copy.deepcopy(SYMM)
    symm["Z2"]["on"] = z2
    config = {
        "print_arch": False,
        "load_model": {"parameters": False, "sampler": False},
        "storage": copy.deepcopy(STORAGE),
        "symm_wrapper": symm,
        "selection": "Test",
        "Test": {"stage0": {"template": {"Sequential": {"ViT2D": {}, "ComplexHead": {}}}}},
    }
    hydra = Hydra(config, lattice_size=LATTICE, hilbert=nk.hilbert.Spin(0.5, N))
    params = hydra.model.init(jax.random.PRNGKey(0), jnp.ones((1, N)))["params"]
    return hydra, params


def schedule_setup(modes, lrs, epochs=None):
    epochs = epochs or [2] * len(modes)
    return {
        "learning_rate": {
            "epochs_struct": [[epochs]],
            "modes_struct": [[modes]],
            "lr_struct": [[lrs]],
            "rescale": 1.0,
        },
        "diag_shift": "lin(0.0001,0.0001,300)",
    }


# ------------------------------------------------------------------ codes


def test_wrapper_level_does_not_consume_a_digit():
    hydra, params = build()
    c2p = hydra.get_code2path(params)
    assert c2p["0"] == ("NN", "Seq_0")
    assert c2p["1"] == ("NN", "Seq_1")
    assert ("NN",) not in c2p.values()
    assert all(path[0] == "NN" for path in c2p.values())


def test_symmetry_wrapper_does_not_consume_a_digit():
    hydra, params = build(z2=True)
    c2p = hydra.get_code2path(params)
    assert c2p["0"] == ("NN", "Z2Wrap", "Seq_0")
    assert c2p["1"] == ("NN", "Z2Wrap", "Seq_1")


def test_legacy_checkpoint_root_key_is_renamed():
    """Checkpoints saved before the refactor have 'SymmModel' at the root:
    on load it becomes 'NN', so paths and codes match the current model."""
    _, params = build()
    legacy = {"SymmModel": params["NN"]}
    upgraded = upgrade_legacy_params(legacy)
    assert list(upgraded) == ["NN"]
    assert get_code2path_flatten(upgraded) == get_code2path_flatten(params)
    # Also inside a {'params': ...} variables dict
    assert list(upgrade_legacy_params({"params": legacy})["params"]) == ["NN"]


def test_printed_tree_uses_same_codes(capsys):
    hydra, params = build()
    print_architecture(get_code2path_tree(params), print_all=False)
    out = capsys.readouterr().out
    assert "(0)" in out and "(1)" in out
    assert "NN: (no symmetrization)" in out
    assert "NN   --->" not in out


# ---------------------------------------------------------------- schedule


def leaf_labels(schedule, params, info):
    from NN_module.schedule.masks import masked_optimizer

    paths = [inf[1] for inf in info]
    tree = masked_optimizer(params, paths, schedule.eon_path2code)
    return {"/".join(p): v for p, v in flatten_with_paths(tree)}


def test_mode_A_with_one_lr_per_first_level_submodule():
    hydra, params = build()
    schedule = Schedule(schedule_setup([["A"]], [[0.01, 0.001]]), hydra.get_code2path(params))
    (lr_funcs, info, _), = list(schedule.schedule())

    labels = leaf_labels(schedule, params, info)
    assert set(labels.values()) == {"train_0", "train_1"}
    assert all(v == "train_0" for k, v in labels.items() if "/Seq_0/" in k)
    assert all(v == "train_1" for k, v in labels.items() if "/Seq_1/" in k)
    assert [inf[2] for inf in info] == [0.01, 0.001]

    opt = schedule.transform_optimizer(params, OPTIMIZER, info, lr_funcs)
    state = opt.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = opt.update(grads, state, params)
    seq0 = flatten_with_paths(updates["NN"]["Seq_0"])
    seq1 = flatten_with_paths(updates["NN"]["Seq_1"])
    assert all(jnp.allclose(u, -0.01) for _, u in seq0)
    assert all(jnp.allclose(u, -0.001) for _, u in seq1)


def test_mode_A_with_single_lr_trains_everything():
    hydra, params = build()
    schedule = Schedule(schedule_setup([["A"]], [[0.01]]), hydra.get_code2path(params))
    (_, info, _), = list(schedule.schedule())
    assert "freeze" not in leaf_labels(schedule, params, info).values()


def test_mode_A_with_wrong_number_of_lrs_gives_clear_error():
    hydra, params = build()
    schedule = Schedule(
        schedule_setup([["A"]], [[0.1, 0.01, 0.001]]), hydra.get_code2path(params)
    )
    with pytest.raises(ValueError, match="3 learning rates"):
        list(schedule.schedule())


def test_single_code_freezes_the_rest():
    hydra, params = build()
    schedule = Schedule(schedule_setup([["1"]], [[0.01]]), hydra.get_code2path(params))
    (_, info, _), = list(schedule.schedule())
    labels = leaf_labels(schedule, params, info)
    assert all(v == "freeze" for k, v in labels.items() if "/Seq_0/" in k)
    assert all(v == "train_1" for k, v in labels.items() if "/Seq_1/" in k)


def test_unknown_mode_code_gives_clear_error():
    hydra, params = build()
    with pytest.raises(KeyError, match="'7' does not exist"):
        Schedule(schedule_setup([["7"]], [[0.01]]), hydra.get_code2path(params))


def test_flat_setup_records_real_modes():
    hydra, params = build()
    schedule = Schedule(
        schedule_setup([["A"], ["1"], ["0", "1"]], [[0.01], [0.01], [0.01, 0.02]]),
        hydra.get_code2path(params),
    )
    assert schedule.flat_setup()["modes"] == [
        ["A"],
        ["NN/Seq_1"],
        ["NN/Seq_0", "NN/Seq_1"],
    ]


# ------------------------------------------------------------- hydra load


def test_load_bad_code_lists_whole_network_code():
    hydra, params = build()
    hydra.load = [["stageX0", "9", "A"]]
    hydra.params_history = {"stageX0": params}
    with pytest.raises(KeyError, match="'A' \\(whole network\\)"):
        hydra.weight_transplantation(params)


def test_load_submodule_into_submodule():
    """'0' -> '0' copies only the first submodule (here the ViT body)."""
    hydra, params = build()
    _, other = build()
    other = jax.tree_util.tree_map(lambda x: x + 1.0, other)
    hydra.load = [["stageX0", "0", "0"]]
    hydra.params_history = {"stageX0": other}
    new = hydra.weight_transplantation(params)
    # Compare by path (tree_map returns dicts with sorted keys)
    fn, fo, fp = (dict(flatten_with_paths(t)) for t in (new, other, params))
    assert all(jnp.array_equal(fn[p], fo[p]) for p in fn if p[1] == "Seq_0")
    assert all(jnp.array_equal(fn[p], fp[p]) for p in fn if p[1] == "Seq_1")


def test_printed_symmetry_label_does_not_accumulate(capsys):
    """print_architecture used a mutable default list and accumulated labels."""
    _, params = build(z2=True)
    capsys.readouterr()
    for _ in range(3):
        print_architecture(get_code2path_tree(params), print_all=False)
    out = capsys.readouterr().out
    assert out.count("NN: (Z2Wrap)") == 3
