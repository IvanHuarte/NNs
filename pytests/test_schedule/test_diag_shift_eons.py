"""
diag_shift: one curve for the whole run (string) or one per eon (list).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_schedule -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.schedule.schedule import Schedule
from Hydra.schedule.utils import period_from_string


def setup(ds, eons=((30, 20), (40,), (10, 10))):
    return {
        "learning_rate": {
            "epochs_struct": [[list(e)] for e in eons],
            "modes_struct": [[[["A"] for _ in e]] for e in eons],
            "lr_struct": [[[[0.01] for _ in e]] for e in eons],
            "rescale": 1.0,
        },
        "sampler": {},
        "diag_shift": ds,
    }


def values(schedule):
    funcs, info = schedule.get_diag_shift_schedule()
    periods = schedule.flat_setup()["epochs"]
    return np.concatenate([[float(f(i)) for i in range(n)] for f, n in zip(funcs, periods)]), info


def test_stage_epochs():
    s = Schedule(setup("lin(0.01,0.001)"), {"0": ()})
    assert s.stage_epochs == [50, 40, 20] and s.total_epochs == 110


def test_single_string_spans_the_run():
    s = Schedule(setup("lin(0.01,0.001)"), {"0": ()})
    v, _ = values(s)
    assert v == pytest.approx(np.asarray(period_from_string(110, "lin(0.01,0.001)")))


def test_one_curve_per_eon():
    ds = ["sig(0.01,0.0001,20)", "lin(0.005,0.001)", "lin(0.002,0.002)"]
    s = Schedule(setup(ds), {"0": ()})
    v, info = values(s)
    ref = np.concatenate([np.asarray(period_from_string(n, d)) for n, d in zip([50, 40, 20], ds)])
    assert v == pytest.approx(ref)
    # each eon restarts its curve
    assert v[50] == pytest.approx(0.005) and v[90] == pytest.approx(0.002)
    assert info[0].startswith("sig(") and info[2].startswith("lin(")


def test_wrong_number_of_curves():
    with pytest.raises(ValueError):
        Schedule(setup(["lin(0.01,0.001)"] * 2), {"0": ()}).get_diag_shift_schedule()


def test_one_curve_per_period():
    # eon 0: floor 1e-3 at constant lr, then down to 1e-4 while lr decays; eon 1 one curve
    ds = [["sig(0.01,0.001,10)", "exp(0.001,0.0001)"], "lin(0.005,0.001)", ["lin(0.002,0.002)"] * 2]
    v, info = values(Schedule(setup(ds), {"0": ()}))
    ref = np.concatenate([np.asarray(period_from_string(n, d)) for n, d in
                          zip([30, 20, 40, 10, 10], [ds[0][0], ds[0][1], ds[1], *ds[2]])])
    assert v == pytest.approx(ref)
    assert v[30] == pytest.approx(0.001) and v[49] < 1.2e-4
    assert info[1].startswith("exp(") and info[2].startswith("lin(")
    with pytest.raises(ValueError):  # 1 curve for an eon with 2 periods
        Schedule(setup([["lin(0.01,0.001)"], "lin(0.01,0.001)", "lin(0.01,0.001)"]), {"0": ()}).get_diag_shift_schedule()
