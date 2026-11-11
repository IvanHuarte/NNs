"""
BestIterKeeper with orbax: several modes at once, one best per mode and
stage on disk, restore into the vstate, Hydra loading from the best tree.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_callbacks -q
"""

import json
import os
import sys
from types import SimpleNamespace

import jax
import netket as nk
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from Hydra.callback.BestIterKeeper import BestIterKeeper
from Hydra.callback.BestIterKeeper.BestIterKeeper import _Tracker
from Hydra.NN.Hydra import resolve_load_path
from Hydra.NN.utils import load_from_file, resolve_best_orbax

N = 8
MODES = ["best_energy", "best_vscore", "best"]


def make_driver(seed=0):
    g = nk.graph.Chain(N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    H = nk.operator.Ising(hi, g, h=1.0)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vstate = nk.vqs.MCState(sampler, nk.models.RBM(alpha=1), n_samples=256, seed=seed)
    driver = nk.driver.VMC_SR(
        H.to_jax_operator(), nk.optimizer.Sgd(0.02), variational_state=vstate, diag_shift=0.01
    )
    return H, vstate, driver


class Recorder:
    def __init__(self):
        self.E, self.var, self.params = [], [], []

    def __call__(self, step, log_data, driver):
        s = log_data[driver._loss_name]
        self.E.append(float(np.real(s.mean)))
        self.var.append(float(np.real(s.variance)))
        self.params.append(jax.device_get(driver.state.parameters))
        return True


def flat(p):
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(p)])


def expected_best(rec, mode, lo, hi, c=8.0, vmax=0.05):
    """Reference selection over steps [lo, hi) (first step always taken)."""
    E, var = np.array(rec.E[lo:hi]), np.array(rec.var[lo:hi])
    V = N * var / E**2
    best = 0
    for i in range(1, len(E)):
        if mode == "best_vscore":
            better = V[i] < V[best]
        elif mode == "best_energy":
            ok_i, ok_b = V[i] < vmax, V[best] < vmax
            better = ok_i if ok_i != ok_b else E[i] < E[best]
        else:
            better = E[i] + abs(E[i]) * V[i] / c < E[best] + abs(E[best]) * V[best] / c
        if better:
            best = i
    return lo + best


def test_modes_on_disk_and_restore(tmp_path):
    H, vstate, driver = make_driver()
    keeper = BestIterKeeper(
        12, H, N, mode=MODES, orbax_path=str(tmp_path / "sim_best.orbax"),
        save_every=3, start_fraction=-1.0,
    )
    rec = Recorder()
    driver.run(12, out=None, callback=[rec, keeper.update], show_progress=False)
    keeper.finalize()

    for mode in MODES:
        step = expected_best(rec, mode, 0, 12)
        tr = keeper.trackers[mode]
        assert tr.best["step"] == step, mode
        path = keeper.best_path(mode, 0)
        # single stage: <root>/<mode>/{parameters, sampler_state, metrics}
        assert path == str(tmp_path / "sim_best.orbax" / mode)
        assert sorted(os.listdir(path)) == ["_CHECKPOINT_METADATA", "metrics", "parameters", "sampler_state"] \
            or {"metrics", "parameters", "sampler_state"} <= set(os.listdir(path))
        assert json.load(open(os.path.join(path, "metrics", "metadata")))["step"] == step
        # restore puts exactly those parameters in the vstate
        keeper.restore(vstate, mode)
        assert np.allclose(flat(vstate.parameters), flat(rec.params[step]))
        assert keeper.best_step == step
        assert keeper.best_state_energy == pytest.approx(rec.E[step], abs=0)

    summary = keeper.summary()
    assert set(summary) == set(MODES) and all(len(v) == 1 for v in summary.values())
    assert summary["best"][0]["score"] == pytest.approx(
        rec.E[summary["best"][0]["step"]]
        + abs(rec.E[summary["best"][0]["step"]]) * summary["best"][0]["vscore"] / 8.0
    )


def test_stages_reset_and_keep_their_own_best(tmp_path):
    H, vstate, driver = make_driver(seed=1)
    keeper = BestIterKeeper(
        10, H, N, mode=["best_energy", "best"], orbax_path=str(tmp_path / "s_best.orbax"),
        save_every=100, start_fraction=-1.0, stage_epochs=[5, 5],
    )
    rec = Recorder()
    driver.run(5, out=None, callback=[rec, keeper.update], show_progress=False)
    keeper.new_stage(1)
    driver.run(5, out=None, callback=[rec, keeper.update], show_progress=False)
    keeper.finalize()

    for mode in ("best_energy", "best"):
        s0 = keeper.summary()[mode][0]
        s1 = keeper.summary()[mode][1]
        assert s0["stage"] == 0 and s1["stage"] == 1
        assert s0["step"] == expected_best(rec, mode, 0, 5)
        assert s1["step"] == expected_best(rec, mode, 5, 10)
        assert 5 <= s1["step"] < 10
        assert s0["path"].endswith(f"{mode}/stage0") and s1["path"].endswith(f"{mode}/stage1")
        assert os.path.isdir(os.path.join(s0["path"], "parameters"))
        assert os.path.isdir(os.path.join(s1["path"], "parameters"))
    # the results use the last stage
    keeper.restore(vstate)
    assert 5 <= keeper.best_step < 10


def test_start_fraction_per_stage():
    """States before start_fraction * stage_epochs are ignored (but the first)."""
    H, vstate, driver = make_driver(seed=2)
    keeper = BestIterKeeper(8, H, N, mode="always", start_fraction=0.5, stage_epochs=[8])
    driver.run(8, out=None, callback=[keeper.update], show_progress=False)
    keeper.finalize()
    assert keeper.trackers["always"].best["step"] == 7


def test_hydra_loads_from_best_tree(tmp_path):
    """A run without results json: load_from (dir) falls back to *_best.orbax."""
    H, vstate, driver = make_driver(seed=3)
    run_dir = tmp_path / "run" / "UUID_x"
    run_dir.mkdir(parents=True)
    keeper = BestIterKeeper(
        6, H, N, mode=["best_vscore", "best"], orbax_path=str(run_dir / "lab_best.orbax"),
        start_fraction=-1.0,
    )
    rec = Recorder()
    driver.run(6, out=None, callback=[rec, keeper.update], show_progress=False)
    keeper.flush()  # no finalize: as if the job had been killed now

    path = resolve_load_path(str(tmp_path / "run"))
    assert path.endswith("lab_best.orbax")
    # "best" is preferred over the other modes
    step_dir = resolve_best_orbax(path)
    assert step_dir.endswith("lab_best.orbax/best")
    params = load_from_file(path, tag="parameters")
    step = keeper.trackers["best"].best["step"]
    assert np.allclose(flat(params), flat(rec.params[step]))
    # explicit mode
    p2 = load_from_file(str(run_dir / "lab_best.orbax" / "best_vscore"), tag="parameters")
    s2 = keeper.trackers["best_vscore"].best["step"]
    assert np.allclose(flat(p2), flat(rec.params[s2]))


@pytest.mark.parametrize(
    "best, cand, expected",
    [
        # small energy rise, big vscore drop -> update
        (dict(energy=-100.0, vscore=0.020), dict(energy=-99.95, vscore=0.010), True),
        # small energy rise, small vscore drop -> keep
        (dict(energy=-100.0, vscore=0.020), dict(energy=-99.95, vscore=0.018), False),
        # big energy drop, small vscore rise -> update
        (dict(energy=-100.0, vscore=0.020), dict(energy=-100.2, vscore=0.025), True),
        # small energy drop, big vscore rise -> keep
        (dict(energy=-100.0, vscore=0.020), dict(energy=-100.05, vscore=0.030), False),
    ],
)
def test_joint_score_tradeoff(best, cand, expected):
    tr = _Tracker("best", {"vscore_max": None, "c": 8.0, "z": 0.0})
    for d in (best, cand):
        d["variance"] = d["vscore"] * d["energy"] ** 2 / 64
        d["n_samples"] = 4096
    tr.best = best
    assert tr.is_better(cand) is expected


def test_best_energy_vscore_bound():
    tr = _Tracker("best_energy", {"vscore_max": 0.05, "c": 8.0, "z": 0.0})
    tr.best = dict(energy=-10.0, vscore=0.2)  # never passed the bound
    assert tr.is_better(dict(energy=-9.0, vscore=0.04))  # passes the bound
    assert tr.is_better(dict(energy=-11.0, vscore=0.3))  # lower energy, none passes
    tr.best = dict(energy=-10.0, vscore=0.04)
    assert not tr.is_better(dict(energy=-11.0, vscore=0.3))
    assert tr.is_better(dict(energy=-10.5, vscore=0.049))


def test_trace_file(tmp_path):
    H, vstate, driver = make_driver(seed=4)
    keeper = BestIterKeeper(7, H, N, mode="best", orbax_path=str(tmp_path / "x_best.orbax"),
                            save_every=3, start_fraction=-1.0, stage_epochs=[4, 3])
    rec = Recorder()
    driver.run(4, out=None, callback=[rec, keeper.update], show_progress=False)
    keeper.new_stage(1)
    driver.run(3, out=None, callback=[rec, keeper.update], show_progress=False)
    keeper.finalize()
    data = np.loadtxt(tmp_path / "x_trace.txt")
    assert data.shape == (7, 6)
    assert list(data[:, 0]) == list(range(7))
    assert list(data[:, 1]) == [0] * 4 + [1] * 3 and list(data[:, 2]) == [0, 1, 2, 3, 0, 1, 2]
    assert np.allclose(data[:, 3], rec.E)


# ---------------------------------------------------------------------------
# Recovery at a stage change: only from a clearly unstable end of stage
# ---------------------------------------------------------------------------


def _fake_stage(keeper, vstate, energies, variances, bad_from=None):
    """Feed the keeper a synthetic stage; parameters are scaled step by step
    so the state of every step is recognisable (θ_t = θ_0 * (1 + t))."""
    p0 = jax.device_get(vstate.parameters)
    for t, (e, v) in enumerate(zip(energies, variances)):
        vstate.parameters = jax.tree_util.tree_map(lambda x: x * (1 + t), p0)
        stats = SimpleNamespace(mean=e, variance=v)
        drv = SimpleNamespace(_loss_name="Energy", state=vstate)
        keeper.update(t, {"Energy": stats}, drv)
    return p0


@pytest.mark.parametrize("case", ["stable", "collapse", "nan", "disabled"])
def test_stage_end_recovery(tmp_path, case):
    H, vstate, _ = make_driver()
    n = 400
    rng = np.random.default_rng(0)
    E = -10 - 2 * (1 - np.exp(-np.arange(n) / 50)) + 0.01 * rng.standard_normal(n)
    var = np.full(n, 0.1)
    if case in ("collapse", "disabled"):
        E[320:] += 1.0  # +8 % energy and 20x variance at the end
        var[320:] = 2.0
    if case == "nan":
        E[-3:] = np.nan
    keeper = BestIterKeeper(
        n, H, N, mode=["best"], orbax_path=str(tmp_path / f"{case}_best.orbax"),
        save_every=1000, start_fraction=0.1, stage_epochs=[n, n],
        recover=False if case == "disabled" else None,
    )
    p0 = _fake_stage(keeper, vstate, E, var)
    out = flat(keeper.stage_end_params(vstate))
    final = flat(vstate.parameters)
    if case in ("stable", "disabled"):
        assert np.allclose(out, final)
    else:
        step = keeper.trackers["best"].best["step"]
        assert step < 320
        assert np.allclose(out, flat(p0) * (1 + step))
    if case == "stable":
        assert keeper.stage_end_check()[0] is False
    keeper.finalize()  # finish the async writes before the interpreter exits


def test_invalid_states_never_kept(tmp_path):
    """Regression (cluster2, Nov 2026): NaN first step kept as best, best_vscore
    on an E=+300 eigenstate-like state, best_energy on a diverged E=-8e59."""
    H, vstate, _ = make_driver()
    keeper = BestIterKeeper(6, H, N, mode=["best", "best_energy", "best_vscore"],
                            orbax_path=str(tmp_path / "inv_best.orbax"), start_fraction=-1.0)
    assert np.isfinite(keeper.E_bound) and keeper.E_bound >= 8  # Ising: sum ||h_k|| = 2N
    E = [np.nan, -10.0, -10.3, 5.0, -1e60, -10.2]
    V = [np.nan, 0.5, 0.4, 1e-6, 1.0, 0.3]  # +5 has the lowest vscore of all
    _fake_stage(keeper, vstate, E, V)
    steps = {m: tr.best["step"] for m, tr in keeper.trackers.items()}
    assert steps["best_energy"] == 2 and steps["best"] in (2, 5)
    assert steps["best_vscore"] in (2, 5)  # within 1 % of the lowest energy
    keeper.finalize()


def test_restore_without_valid_state(tmp_path):
    H, vstate, _ = make_driver()
    keeper = BestIterKeeper(3, H, N, mode="best", orbax_path=str(tmp_path / "nan_best.orbax"))
    _fake_stage(keeper, vstate, [np.nan] * 3, [np.nan] * 3)
    keeper.finalize()
    assert keeper.restore(vstate) is vstate and np.isnan(keeper.best_state_energy)
