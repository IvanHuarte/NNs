"""
Callbacks take the energy from the driver's log_data instead of calling
vstate.expect(H) again (same samples and parameters -> identical value, one
network evaluation on all connected configurations less per step).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_callbacks -q
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import netket as nk
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from NN_module.callback.BestIterKeeper import BestIterKeeper
from NN_module.callback.Checkpoint.Checkpoint import Checkpoint
from NN_module.callback.EnergyPlotter import EnergyPlotter

N = 8
N_STEPS = 4


def make_driver():
    g = nk.graph.Chain(N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    H = nk.operator.Ising(hi, g, h=1.0)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vstate = nk.vqs.MCState(sampler, nk.models.RBM(alpha=1), n_samples=256, seed=0)
    driver = nk.driver.VMC_SR(
        H.to_jax_operator(),
        nk.optimizer.Sgd(0.01),
        variational_state=vstate,
        diag_shift=0.01,
    )
    return H, vstate, driver


class EnergyRecorder:
    """Reference: energy the driver logged at each step."""

    def __init__(self):
        self.energies = []

    def __call__(self, step, log_data, driver):
        self.energies.append(float(np.real(log_data[driver._loss_name].mean)))
        return True


@pytest.mark.parametrize("mode", ["best_energy", "best_vscore", "always"])
def test_callbacks_use_log_data_energy(tmp_path, monkeypatch, mode):
    H, vstate, driver = make_driver()

    # Any extra vstate.expect from a callback would show up here
    calls = []
    orig_expect = type(vstate).expect

    def counting_expect(self, op, *a, **k):
        calls.append(op)
        return orig_expect(self, op, *a, **k)

    monkeypatch.setattr(type(vstate), "expect", counting_expect)

    keeper = BestIterKeeper(N_STEPS, H, N, mode=mode, start_fraction=-1.0)
    plotter = EnergyPlotter(H, N, E_ED=-10.0)
    ckpt = Checkpoint(H, str(tmp_path / "sim"), {"do_each": 1, "async": False})
    rec = EnergyRecorder()

    driver.run(
        N_STEPS,
        out=None,
        callback=[rec, keeper.update, plotter, ckpt],
        show_progress=False,
    )
    ckpt.wait()
    keeper.finalize()
    keeper.restore(vstate)

    assert calls == [], f"callbacks called vstate.expect {len(calls)} times"
    assert len(rec.energies) == N_STEPS
    assert plotter.energies == pytest.approx(rec.energies, abs=0)
    assert keeper.best_state_energy == pytest.approx(
        rec.energies[keeper.best_step], abs=0
    )
    if mode == "always":
        assert keeper.best_step == N_STEPS - 1
    saved = ckpt.manager.restore(N_STEPS - 1)
    assert saved["metrics"]["energy"] == pytest.approx(rec.energies[-1], abs=0)


def test_log_energy_equals_expect():
    """The premise: in callbacks, log_data energy == vstate.expect(H) exactly."""
    H, vstate, driver = make_driver()
    diffs = []

    def cb(step, log_data, driver):
        e_log = log_data[driver._loss_name].mean
        e_exp = driver.state.expect(H).mean
        diffs.append(abs(complex(e_log) - complex(e_exp)))
        return True

    driver.run(N_STEPS, out=None, callback=cb, show_progress=False)
    assert max(diffs) < 1e-10
