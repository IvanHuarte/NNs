"""
Checkpoint (asynchronous writes), SanityMonitor (generic metrics, both
displays), LeavesGradient (module tree, heat map), EnergyPlotter and
ModPhasePlotter (headless, PNG output) on a small RBM run.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_callbacks/test_monitoring_callbacks.py -q
"""

import json
import os
import sys

import flax.linen as nn
import jax
import matplotlib
import netket as nk
import numpy as np
import pytest

matplotlib.use("Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from Hydra.callback import Checkpoint, EnergyPlotter, LeavesGradient, ModPhasePlotter, SanityMonitor
from Hydra.callback.introspection import aggregate, leaf_table, module_of, path_codes
from Hydra.callback.SanityMonitor.metrics import METRICS
from Hydra.NN.utils import load_from_file

N = 8


class TwoBlocks(nn.Module):
    """Tiny network with two named submodules (as Hydra's Seq_0 / Seq_1)."""

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(6, name="Seq_0", param_dtype=np.float64)(x)
        h = nn.tanh(h)
        re = nn.Dense(1, name="Seq_1", param_dtype=np.float64)(h)[..., 0]
        im = nn.Dense(1, name="Seq_2", param_dtype=np.float64)(h)[..., 0]
        return re + 1j * im


def make_driver(seed=0, sz=0, model=None):
    g = nk.graph.Chain(N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=sz)
    H = nk.operator.Heisenberg(hi, g)
    sampler = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model or TwoBlocks(), n_samples=128, seed=seed)
    driver = nk.driver.VMC_SR(H.to_jax_operator(), nk.optimizer.Sgd(0.02), variational_state=vstate,
                              diag_shift=0.01)
    return H, vstate, driver


class Recorder:
    def __init__(self):
        self.params = []

    def __call__(self, step, log_data, driver):
        self.params.append(jax.device_get(driver.state.parameters))
        return True


def flat(p):
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(p)])


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------


@pytest.mark.parametrize("asyn", [True, False])
def test_checkpoint_async_complete_and_loadable(tmp_path, asyn):
    H, vstate, driver = make_driver()
    ck = Checkpoint(H, str(tmp_path / "sim"), {"do_each": 5, "async": asyn, "stage_end": True},
                    stage_epochs=[12])
    rec = Recorder()
    driver.run(12, out=None, callback=[rec, ck], show_progress=False)
    ck.finalize()
    root = tmp_path / "sim_checkpoint.orbax"
    steps = sorted(int(d) for d in os.listdir(root) if d.isdigit())
    assert steps == [0, 5, 10, 11]  # every 5 + last step of the stage
    assert not [d for d in os.listdir(root) if "tmp" in d]
    m = json.load(open(root / "10" / "metrics" / "metrics"))
    assert {"energy", "energy_error", "variance", "vscore", "stage", "local_step"} <= set(m)
    p = load_from_file(str(root), checkpoint=10, tag="parameters")
    assert np.allclose(flat(p), flat(rec.params[10]))


def test_checkpoint_max_to_keep(tmp_path):
    H, vstate, driver = make_driver()
    ck = Checkpoint(H, str(tmp_path / "s"), {"do_each": 2, "async": True, "max_to_keep": 2,
                                             "stage_end": False})
    driver.run(10, out=None, callback=[ck], show_progress=False)
    ck.finalize()
    steps = sorted(int(d) for d in os.listdir(tmp_path / "s_checkpoint.orbax") if d.isdigit())
    assert steps == [6, 8]


def test_checkpoint_stages(tmp_path):
    H, vstate, driver = make_driver()
    ck = Checkpoint(H, str(tmp_path / "t"), {"do_each": 100, "async": True}, stage_epochs=[4, 4])
    driver.run(4, out=None, callback=[ck], show_progress=False)
    ck.new_stage(1)
    driver.run(4, out=None, callback=[ck], show_progress=False)
    ck.finalize()
    root = tmp_path / "t_checkpoint.orbax"
    assert sorted(int(d) for d in os.listdir(root) if d.isdigit()) == [0, 3, 7]
    assert json.load(open(root / "7" / "metrics" / "metrics"))["stage"] == 1


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------


def test_module_grouping_and_shares():
    _, vstate, driver = make_driver()
    params = jax.device_get(vstate.parameters)
    dp = jax.tree_util.tree_map(np.ones_like, params)
    codes = path_codes(params)
    mods, total = aggregate(leaf_table(params, dp), 1, codes)
    labels = [m["label"] for m in mods]
    assert labels == ["0 Seq_0", "1 Seq_1", "2 Seq_2"]
    assert sum(m["share"] for m in mods) == pytest.approx(1.0)
    assert total["size"] == sum(np.size(x) for x in jax.tree_util.tree_leaves(params))
    # wrapper labels do not count as modules
    assert module_of(("params", "NN", "Z2Wrap", "SeqV_0", "Dense_0", "kernel"), 1) == ("SeqV_0",)


# --------------------------------------------------------------------------
# SanityMonitor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["rich", "simple"])
@pytest.mark.parametrize("verbose", [0, 1, 2, 3, 4])
def test_sanity_monitor_all_metrics(tmp_path, capsys, mode, verbose):
    H, vstate, driver = make_driver()
    log = tmp_path / "sim_sanity.jsonl"
    sm = SanityMonitor({"do_each": 3, "mode": mode, "verbose": verbose, "n_jacobian": 32,
                        "metrics": {k: True for k in METRICS}}, N=N, E_ref=-15.0, log_path=str(log))
    driver.run(7, out=None, callback=[sm], show_progress=False)
    out = capsys.readouterr().out
    recs = [json.loads(line) for line in open(log)]
    assert [r["step"] for r in recs] == [3, 6]
    assert set(recs[0]["metrics"]) == set(METRICS)
    assert recs[1]["metrics"]["energy"]["dE"] is not None  # trend w.r.t. the previous check
    assert recs[0]["overall"] in ("ok", "warn", "error")
    if verbose == 0:
        assert "SANITY MONITOR" not in out
    elif verbose >= 3:
        assert "SANITY MONITOR" in out


def test_sanity_monitor_old_config_names():
    sm = SanityMonitor({"do_each": 10, "mode": "simple", "verbose": 1,
                        "metrics": {"gradients": True, "sampling": True, "sample_histogram": True,
                                    "phase": True}})
    assert sm.selection == ["gradients", "sampling", "magnetization", "wavefunction"]
    with pytest.raises(ValueError):
        SanityMonitor({"metrics": {"nope": True}})


# --------------------------------------------------------------------------
# LeavesGradient
# --------------------------------------------------------------------------


def test_leaves_gradient_tree_and_heatmap(tmp_path, capsys):
    H, vstate, driver = make_driver()
    lg = LeavesGradient({"do_each": 2, "depth": 1, "n_jacobian": 16}, log_prefix=str(tmp_path / "sim"))
    driver.run(6, out=None, callback=[lg], show_progress=False)
    lg.new_stage(1)
    driver.run(3, out=None, callback=[lg], show_progress=False)
    lg.finalize()
    out = capsys.readouterr().out
    assert "LeavesGradient" in out and "Seq_0" in out
    assert (tmp_path / "sim_leaves_heatmap.png").is_file()
    d = np.load(tmp_path / "sim_leaves.npz")
    assert d["log10_rel"].shape == (len(d["leaves"]), len(d["steps"]))


# --------------------------------------------------------------------------
# Plotters (headless)
# --------------------------------------------------------------------------


def test_energy_plotter_png(tmp_path):
    H, vstate, driver = make_driver()
    ep = EnergyPlotter(N, {"savefig_each": 4, "window": 3, "interactive": False}, E_ref=-14.0, E_global=-15.0,
                       out_path=str(tmp_path / "e.png"))
    driver.run(6, out=None, callback=[ep], show_progress=False)
    ep.new_stage(1)
    driver.run(3, out=None, callback=[ep], show_progress=False)
    ep.finalize()
    assert (tmp_path / "e.png").is_file() and len(ep.E) == 9


def test_energy_plotter_disabled_without_output():
    ep = EnergyPlotter(N, {"interactive": False}, out_path=None)
    assert not ep.enabled
    assert ep(0, None, None) is True


def test_modphase_exact_with_ed_alignment(tmp_path):
    H, vstate, driver = make_driver()
    # "ED" = the network state itself times a global phase: fidelity 1 after alignment
    x = np.asarray(vstate.to_array()) * np.exp(1j * 0.7)
    mp = ModPhasePlotter({"plot_each": 2, "savefig_each": 2, "interactive": False}, x_ED=x, out_path=str(tmp_path / "m.png"))
    mp(0, None, driver)
    assert mp.mode == "exact"
    assert "fidelidad |⟨ψ_ED|ψ⟩| = 1.00000" in mp.ax_m.get_title()
    driver.run(3, out=None, callback=[mp], show_progress=False)
    mp.finalize()
    assert (tmp_path / "m.png").is_file()


def test_modphase_sample_mode(tmp_path):
    H, vstate, driver = make_driver()
    mp = ModPhasePlotter({"plot_each": 1, "max_states": 10, "interactive": False}, out_path=str(tmp_path / "m.png"))
    driver.run(2, out=None, callback=[mp], show_progress=False)
    mp.finalize()
    assert mp.mode == "samples" and (tmp_path / "m.png").is_file()
