"""
Regression tests for path-code / load-path / mask / change_module_attr fixes.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_audit_fixes -q
"""

import os
import sys

import jax.numpy as jnp
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN.Hydra import resolve_load_path
from Hydra.NN.utils import (
    change_module_attr,
    get_code2path_flatten,
    get_code2path_tree,
)
from Hydra.schedule.masks import train_branches


def _wide(n):
    return {f"M_{i:02d}": {"Dense_0": {"kernel": jnp.ones(1)}} for i in range(n)}


def test_more_than_ten_children_codes_do_not_collide():
    c2p = get_code2path_flatten(_wide(12))
    assert c2p["b"] == ("M_11",)
    assert c2p["10"] == ("M_01", "Dense_0")
    assert len({v for v in c2p.values()}) == len(c2p) == 24
    with pytest.raises(ValueError):
        get_code2path_flatten(_wide(37))


def test_codes_independent_of_key_order():
    a = {"Seq_0": {"k": jnp.ones(1)}, "Dense_0": {"k": jnp.ones(1)}}
    b = dict(reversed(list(a.items())))
    assert get_code2path_flatten(a) == get_code2path_flatten(b) == {
        "0": ("Dense_0",), "1": ("Seq_0",)}
    assert get_code2path_tree(a)["Dense_0"]["_idx"] == "0"


def test_orbax_paths_returned_unchanged(tmp_path):
    ck = tmp_path / "run.orbax"
    ck.mkdir()
    (tmp_path / "x_results.json").write_text("{}")
    assert resolve_load_path(str(ck)) == str(ck)
    assert resolve_load_path(str(ck) + "/") == str(ck) + "/"
    assert resolve_load_path(str(ck / "sub")) == str(ck / "sub")


def test_nested_modes_deepest_wins():
    p2c = {("A",): "0", ("A", "B"): "00"}
    f = train_branches([("A",), ("A", "B")], p2c)
    assert f(("A", "B", "kernel"), None) == "train_00"
    assert f(("A", "C", "kernel"), None) == "train_0"
    assert f(("Z", "kernel"), None) == "freeze"


def test_change_attr_recurses_into_changed_wrapper():
    setup = {"module": "Transversal", "setup": {
        "depth": 1, "branch": {"module": "Transversal", "setup": {"depth": 1}}}}
    new, _ = change_module_attr(setup, {"Transversal": {"depth": 3}})
    assert new["setup"]["depth"] == 3
    assert new["setup"]["branch"]["setup"]["depth"] == 3
