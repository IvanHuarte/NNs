"""aplot: index, --where syntax, --app/--for, the three figure kinds, and the old autoplot fixes."""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from autoplots import index as idx  # noqa: E402
from autoplots.aplot import main  # noqa: E402
from autoplots.plots.energyVSirrep import full_irreps  # noqa: E402
from autoplots.utils import get_criterion, group_by, natural_key  # noqa: E402


def nn_tree(net, irrep, odd):
    seq = {"module": "Sequivariant", "setup": {
        "SeqV_0": {"module": net, "setup": {}},
        "SeqV_1": {"module": "OutputHead", "setup": {}},
        "irrep": list(irrep)}}
    return {"module": "NNWrapper", "setup": {"module": "Z2", "setup": {"Wrap": seq, "trivial_Z2": not odd}}}


def fake_run(root, job, L, J2, net, irrep, odd, E, vscore=0.01, dtype="float32", trace=True):
    folder = root / job / "J1J2Square" / "m" / f"Size_{L}x{L}"
    folder.mkdir(parents=True)
    N = L * L
    art = {
        "CM": {"name": "J1J2Square", "size": [L, L], "params": {"J1": 1.0, "J2": J2}, "S_operators": False},
        "NN": {"name": f"{net}OH{irrep[0]}{irrep[1]}{int(odd)}_stage0", "setup": nn_tree(net, irrep, odd)},
        "SIM": {"network_dtype": dtype, "seed": 1, "sampler": {"nsamples": 1024},
                "schedule": {"total_epochs": 100}},
        "results": {"E_best": E, "E_best_per_site": E / N, "vscore": vscore, "time_exe": 360.0,
                    "mssg": "Simulation finished with success", "energy_units": "Pauli",
                    "E_gr_irrep": None, "best_step": 90},
        "metadata": {"uuid": job, "date": "11/11/26"},
    }
    path = folder / f"{job}_results_{L}x{L}.json"
    path.write_text(json.dumps(art))
    if trace:
        steps = np.arange(100)
        e = E + 5 * np.exp(-steps / 10)
        e[60] = -1e40  # one diverged sample: must not become E50
        d = np.c_[steps, 0 * steps, steps, e, 0 * e + 1.0, 0 * e + vscore]
        np.savetxt(folder / f"{job}_trace.txt", d, header="step stage local_step energy variance vscore")
    return path


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "CACHE", tmp_path / "cache" / "index.pkl")
    monkeypatch.setattr(idx, "REFS", tmp_path / "refs.json")
    root = tmp_path / "sims"
    fake_run(root, "a", 8, 0.5, "CvT3", (0, 0), False, -127.5)
    fake_run(root, "b", 8, 0.5, "CvT3", (0, 0), True, -126.0, vscore=0.2)
    fake_run(root, "c", 10, 0.5, "CvT3", (0, 0), False, -198.8)
    fake_run(root, "d", 8, 0.4, "ViT2D", (4, 4), True, -128.0, dtype="float64", trace=False)
    fake_run(root, "e", 10, 0.5, "CvT3", (0, 0), False, -198.5)  # duplicate point of c
    return root


def test_index_columns(runs):
    df = idx.load_index([str(runs)], verbose=False).set_index("job")
    assert df.loc["a", "L"] == 8 and df.loc["a", "N"] == 64
    assert df.loc["b", "irrep"] == "001" and df.loc["d", "irrep"] == "441"
    assert df.loc["a", "net"] == "CvT3" and df.loc["a", "head"] == "OH"
    assert df.loc["a", "E_site_spin"] == pytest.approx(-127.5 / 4 / 64)
    assert df.loc["a", "E50"] > -128 and df.loc["a", "E50"] == pytest.approx(-127.5, abs=0.2)
    assert np.isnan(df.loc["d", "E50"])
    assert df.loc["a", "s_it"] == pytest.approx(3.6)
    # cached second load gives the same table
    again = idx.load_index([str(runs)], verbose=False).set_index("job")
    assert again.loc["c", "E"] == df.loc["c", "E"]


def test_where_syntax(runs):
    df = idx.load_index([str(runs)], verbose=False)
    jobs = lambda w: sorted(idx.select(df, w)["job"])
    assert jobs("net='CvT3' and dtype=='float32' and not vscore>0.1") == ["a", "c", "e"]
    assert jobs("irrep==001") == ["b"]
    assert jobs("name~='ViT'") == ["d"]
    assert jobs("L in [10] and J2==0.5") == ["c", "e"]
    assert jobs("seed==1 and nsamples>=1024") == ["a", "b", "c", "d", "e"]  # read from the JSONs
    with pytest.raises(KeyError):
        idx.select(df, "no_such_key==1")


def test_gap_and_refs(runs, tmp_path):
    (tmp_path / "refs.json").write_text(json.dumps(
        {"J1J2Square": [{"L": 10, "J2": 0.5, "irrep": "000", "E_site_spin": -0.5, "source": "test"}]}))
    df = idx.load_index([str(runs)], verbose=False)
    sel = idx.ensure(idx.select(df, "L==8 and J2==0.5"), ["gap"]).set_index("job")
    assert sel.loc["a", "gap"] == 0 and sel.loc["b", "gap"] == pytest.approx(1.5 / 4)
    c = df.set_index("job").loc["c"]
    assert c["E_ref"] == pytest.approx(-200.0) and c["err_rel"] == pytest.approx(0.006)


def test_xy_app_for(runs, tmp_path):
    out = tmp_path / "fig"
    main(["y=E_site_spin", "x=L", "hue=irrep", "--where", "net='CvT3'", "--app", "--where", "net='ViT2D'",
          "-d", str(runs), "-o", str(out), "--name", "t"])
    data = (out / "t.csv").read_text().splitlines()
    assert data[0].startswith("# aplot ") and (out / "t.png").exists()
    assert len(data) == 2 + 4  # header lines + (L8 000, L10 000 best of c/e, L8 001, ViT L8 441)
    assert "c_results" in "".join(data) and "e_results" not in "".join(data)  # agg=best keeps the lowest E
    main(["y=E", "x=J2", "fit=lin", "--for", "L", "-d", str(runs), "-o", str(out), "--name", "f"])
    assert (out / "f_L8.png").exists() and (out / "f_L10.png").exists()


def test_trace_and_table(runs, tmp_path):
    out = tmp_path / "fig"
    main(["trace", "y=E_site_spin", "smooth=5", "--where", "L==8", "-d", str(runs), "-o", str(out), "--name", "tr"])
    assert (out / "tr.png").exists()
    main(["table", "cols=job,E,E50", "sort=E", "-d", str(runs), "-o", str(out), "--name", "tb"])
    lines = (out / "tb.md").read_text().splitlines()
    assert lines[3].startswith("| c |")  # lowest energy first


def test_old_autoplot_fixes(tmp_path):
    assert sorted([10, 8, -0.1, -0.2], key=natural_key) == [-0.2, -0.1, 8, 10]
    assert sorted(["c10", "c8"], key=natural_key) == ["c8", "c10"]
    paths = []
    for L in (10, 8, 12):
        p = tmp_path / f"{L}.json"
        p.write_text(json.dumps({"CM": {"size": [L, L]}}))
        paths.append(str(p))
    assert list(group_by(paths, "size")) == ["[8, 8]", "[10, 10]", "[12, 12]"]
    assert get_criterion({"results": {"fidelity_irrep": None, "fidelity_global": 0.9}}, "E_best") == "fidelity_global"
    assert get_criterion({"results": {"fidelity_irrep": None}}, "E_best") == "E_best"
    assert len(full_irreps("irreps_Tx_Ty_", (4, 4))) == 16
    assert len(full_irreps("irreps_Tx_Ty_Z2", (4, 4))) == 32
