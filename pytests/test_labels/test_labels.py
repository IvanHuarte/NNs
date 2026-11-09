"""
Labels, folders, console summary and callback plots.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_labels -q
"""

import json
import re
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from rich.console import Console

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

import _label_utils_legacy as legacy

from NN_module.callback.utils import (
    TrainingHistory,
    checkpoint_callback,
    dump_callback,
)
from NN_module.display import display_simulation_settings
from NN_module.exact_diagonalization import EDResult
from NN_module.label_utils import (
    labels_from_artifact,
    make_labels,
    make_run_paths,
    make_title,
)
from NN_module.schedule.schedule import Schedule
from NN_module.schedule.utils import get_schedule_label

DATE = "20261106T101010"
UUID = "abcd1234"

CM_SETUPS = [
    {"name": "J1J2Square", "size": [4, 4], "params": {"J1": 1.0, "J2": 0.5, "fields": [0.0, 0.1, 0.0]}},
    {"name": "Oxalate", "size": [4, 4], "params": {"couplings": [0.2, 9.0, 72.0]}},
    {"name": "LRChain", "size": [8, 1], "params": {"J": 1.0, "alpha": 3.0, "fields": [0.5, 0.0]}},
    {"name": "LRSquare", "size": [4, 4], "params": {"J": -1.0, "alpha": 1.5, "fields": [0.5, 0.2]}},
    {"name": "SSMSquare", "size": [4, 4], "params": {"J": 1.0, "Jp": 0.7, "fields": [0.0, 0.0, 0.0]}},
    {
        "name": "IsingSquare",
        "size": [4, 4],
        "params": {"fields": [0.5, 0.1], "J": -1.0, "ops": [["X", "Z"], ["ZZ"]]},
    },
]


@pytest.fixture
def frozen_legacy_date(monkeypatch):
    class FakeDatetime:
        @staticmethod
        def now():
            return SimpleNamespace(strftime=lambda fmt: DATE)

    monkeypatch.setattr(legacy, "datetime", FakeDatetime)


@pytest.mark.parametrize("cm", CM_SETUPS, ids=lambda c: c["name"])
def test_labels_identical_to_legacy(cm, frozen_legacy_date):
    """File names must not change: analysis scripts parse them."""
    old = legacy.get_filenames_from_settings(cm, {"name": "Net", "setup": {}}, UUID)
    new = make_labels(cm, "Net", UUID, date=DATE)
    # Title: J1J2/SSM used '%1.f' (no decimals) for a field component: fixed
    if cm["name"] in ("J1J2Square", "SSMSquare"):
        assert (new.sim, new.ED, new.json) == old[:3]
        assert new.title.startswith("Callback  " + cm["name"] + "_Net  ")
        fields = ",".join(f"{v:.1f}" for v in cm["params"]["fields"])
        assert f"XYZ=({fields})" in new.title
    else:
        assert (new.sim, new.ED, new.json, new.title) == old


def test_j1j2_label_is_parsed_by_plot_scripts():
    import re

    lab = make_labels(CM_SETUPS[0], "Net", UUID, date=DATE)
    assert "_results_" in lab.json and "UUID_" in lab.json
    J1, J2 = re.search(r"_J1J2_([0-9\.]+_[0-9\.]+)_", lab.json).group(1).split("_")
    assert (float(J1), float(J2)) == (1.0, 0.5)


def test_unregistered_model_gets_generic_label():
    cm = {"name": "NewModel", "size": [3, 3], "params": {"g": 0.3, "h": [1, 2], "g_list": [1]}}
    lab = make_labels(cm, "Net", UUID, date=DATE)
    assert lab.sim == f"NewModel_Net_simulation_3x3_g_0.3_h_1_2_date_{DATE}_UUID_{UUID}"
    assert "g=0.3" in lab.title and "g_list" not in lab.title


def test_oxalate_label_before_preprocess():
    raw = {"name": "Oxalate", "size": [4, 4], "params": {"strength": 0.2, "theta": 9.0, "phi": 72.0}}
    assert make_labels(raw, "N", UUID, DATE) == make_labels(CM_SETUPS[1], "N", UUID, DATE)


def test_labels_from_artifact():
    lab = make_labels(CM_SETUPS[0], "Net", UUID, date=DATE)
    art = {"CM": CM_SETUPS[0], "NN": {"name": "Net"}, "metadata": {"uuid": UUID}}

    # Stored labels win
    assert labels_from_artifact({**art, "labels": lab.as_dict()}) == lab
    # Legacy artifact: rebuilt with the date of its file name
    assert labels_from_artifact(art, f"/x/{lab.json}.json") == lab
    assert make_title(CM_SETUPS[0], "Net") == lab.title


def test_run_paths_same_layout():
    config = json.load(open(os.path.join(ROOT, "config0.json")))
    config["write_folder_sim"] = "/base/"
    nn_setup = {"stage0": {"template": {"Sequential": {"ViT2D": {}, "OutputHead": {}}}}}
    paths = make_run_paths(config, "J1J2Square", "Net", nn_setup, [6, 6], sim_uuid=UUID)

    schedule_label = get_schedule_label(config["schedule"]["learning_rate"])
    assert paths.write_folder == (
        "/base/J1J2Square/Net_Sequential_ViT2D_OutputHead/Size_6x6/"
        f"{schedule_label}/UUID_{UUID}/"
    )
    assert paths.parent == f"/base/J1J2Square/Net_Sequential_ViT2D_OutputHead/Size_6x6/{schedule_label}/"
    assert len(make_run_paths(config, "A", "B", nn_setup, [2, 2]).uuid) == 8


# ============================================================
# Console summary
# ============================================================


def test_display_simulation_settings():
    from NN_module.sampler.sampler import resolve_sampler_setup

    c0 = json.load(open(os.path.join(ROOT, "config0.json")))
    c2 = json.load(open(os.path.join(ROOT, "config2_Hydra.json")))
    nn_name = c2["selection"]
    schedule = Schedule(c0["schedule"], {"0": ("NN", "a"), "1": ("NN", "b")})
    _, ds_info = schedule.get_diag_shift_schedule()

    console = Console(record=True, width=160)
    display_simulation_settings(
        cm_setup={**CM_SETUPS[0], "kwargs_lattice": {"bc": "periodic"}, "S_operators": False},
        nn_name=nn_name,
        nn_evol_setup=c2[nn_name],
        sampler_setup=resolve_sampler_setup(c0["sampler"]),
        n_samples=2048,
        n_ranks=1,
        schedule=schedule,
        ds_info=ds_info,
        optimizer_setup=c0["optimizer"],
        vmc_setup=c0["VMC"],
        console=console,
    )
    out = console.export_text()
    for text in ("n_samples", "2048", "n_ranks", "Schedule", "diag_shift",
                 "J2", "0.5", "periodic", c0["VMC"]["selection"], "stage0"):
        assert text in out, text
    assert "epochs_struct" not in out  # no raw structs


# ============================================================
# Callback plots
# ============================================================


def fake_log(n=30, E0=-10.0):
    E = E0 + np.exp(-np.arange(n) / 5.0)
    return {
        "Energy": {
            "Mean": E + 0j,
            "Sigma": 0.1 * np.ones(n),
            "Variance": 0.01 * np.ones(n),
        }
    }


@pytest.mark.parametrize("with_ed", [True, False])
def test_dump_callback(tmp_path, with_ed):
    ed = EDResult(E_global=-10.5, E_irrep=-10.2, irrep=[0, 0]) if with_ed else EDResult()
    lab = make_labels(CM_SETUPS[0], "Net", UUID, date=DATE)
    log = fake_log()
    art = dump_callback(
        log,
        ed=ed,
        labels=lab,
        write_folder=str(tmp_path) + "/",
        size=[4, 4],
        time_exe=1.0,
        best_step=12,
        best_energy=float(log["Energy"]["Mean"][12].real),
        do_each_checkpoint=10,
        message="done",
    )
    assert os.path.isfile(art["plot"])
    assert ("error" in art) == with_ed
    if with_ed:  # error w.r.t. the irrep energy
        err = np.loadtxt(art["error"])
        E = log["Energy"]["Mean"].real
        np.testing.assert_allclose(err, np.abs(E + 10.2) / 10.2)


def test_training_history_vscore():
    hist = TrainingHistory.from_log(fake_log(), N=16)
    np.testing.assert_allclose(hist.vscore, 16 * 0.01 / hist.energy**2)


@pytest.mark.parametrize("legacy_units", [True, False])
def test_checkpoint_callback(legacy_units):
    lab = make_labels(CM_SETUPS[0], "Net", UUID, date=DATE)
    results = {
        "E_gr_global": -2.5 if legacy_units else -10.0,
        "E_gr_irrep": None,
        "E_best": -2.4 if legacy_units else -9.6,
        "time_exe": 3.0,
        "best_step": 20,
        "irrep_info": None,
    }
    if not legacy_units:
        results["energy_units"] = "Pauli"
    artifact = {
        "CM": CM_SETUPS[0],
        "NN": {"name": "Net"},
        "SIM": {"schedule": {"total_epochs": 40}},
        "results": results,
        "labels": lab.as_dict(),
    }
    steps, energy = [0, 10, 20, 30], [-8.0, -9.0, -9.5, -9.6]
    fig, ax = checkpoint_callback(artifact, steps, energy, [1e-1, 1e-2, 1e-3, 1e-3])
    assert len(ax) == 3
    # Reference line at -10 in both cases (legacy energies stored as E/4)
    ref = ax[0].collections[0].get_segments()[0][0][1]
    assert ref == pytest.approx(-10.0)
    np.testing.assert_allclose(ax[1].lines[0].get_ydata(), np.abs(np.array(energy) + 10) / 10)

    fig, ax = checkpoint_callback(None, steps, energy, [1, 1, 1, 1])
    assert len(ax) == 2


# ------------------------------------------------------------------
# Architecture label and experiment names
# ------------------------------------------------------------------

from NN_module.label_utils import arch_label, experiment_name, irrep_code

def tpl(net, irrep, coset=(2, 2), **net_kw):
    return {"Sequivariant": {net: dict(net_kw), "OutputHead": {}, "irrep": list(irrep),
                             "phase_mode": "angle", "coset": list(coset)}}


def test_experiment_name():
    assert experiment_name("CvT3", "OutputHead", (2, 2), 1) == "CvT3OH221"
    assert experiment_name("ViT2D", "OutputHead", (0, 0), 0, "b") == "ViT2DOH000b"
    assert irrep_code((10, 0), 1) == "10.0.1"


def test_arch_label_single_stage():
    evo = {"stage0": {"template": tpl("CvT3", (4, 4), kernel=[[3, 3], [3, 3]])}}
    # class names only: no irrep, coset or hyperparameters
    assert arch_label(evo) == "Sequivariant_CvT3-OutputHead"


def test_arch_label_irrep_chain():
    evo = {
        "stage0": {"template": tpl("ViT2D", (0, 0))},
        "stage1": {"template": tpl("ViT2D", (4, 4)), "symm_wrapper": {"Z2": {"trivial_Z2": False}}},
        "stage2": {"template": {"Sequential": {"ViT2D": {}, "OutputHead": {}}}},
    }
    # repeated architectures are written once
    assert arch_label(evo) == "Sequivariant_ViT2D-OutputHead+Sequential_ViT2D-OutputHead"


def test_arch_label_nested_flux():
    evo = {"stage0": {"template": {"Sequential": {
        "Transversal": {"CNN": {}, "MLP": {}}, "OutputHead": {}}}}}
    assert arch_label(evo) == "Sequential_(Transversal_CNN-MLP)-OutputHead"


def test_make_labels_with_arch():
    lab = make_labels(CM_SETUPS[0], "CvT3OH000", UUID, date=DATE, arch="CvT3-x_OH_k000")
    assert lab.sim.startswith("J1J2Square_CvT3OH000_CvT3-x_OH_k000_simulation_4x4_")
    assert lab.json.startswith("J1J2Square_CvT3OH000_CvT3-x_OH_k000_results_4x4_")
    assert lab.ED.startswith("J1J2Square_CvT3OH000_xED_4x4_")
    J1, J2 = re.search(r"_J1J2_([0-9\.]+_[0-9\.]+)_", lab.json).group(1).split("_")
    assert (J1, J2) == ("1.0", "0.5")
    assert make_labels(CM_SETUPS[0], "N", UUID, DATE) == make_labels(CM_SETUPS[0], "N", UUID, DATE, arch=None)
