#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python
"""
Re-evaluate the keeper best state of finished runs with many samples.

E_best in the results JSON is the lowest of many noisy training estimates,
so it is biased low. This loads <sim>_best.orbax/<mode>, samples
--samples configurations from the stored chains and writes, next to the
JSON, <json>_reeval.json = {mode: {E, E_err, var, vscore, tau, R_hat,
n_samples, folder, date}}. The index reads it as E_re, E_re_err, var_re,
vscore_re, tau_re, R_hat_re (keeper primary mode).

    autoplots/reeval.py -d Simulations_claude_cesga --where "L==8 and status=='ok'"
    autoplots/reeval.py RUN_results.json ... [--samples 65536] [--mode best] [--chunk 1024] [--force]

Runs on the device JAX picks (GPU on the clusters); JAX_PLATFORMS=cpu for small lattices.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoplots.index import load_index, select, sidecar  # noqa: E402


def local(path, json_path):
    """Artifact path as written by the run, or the same name next to the JSON (copied results)."""
    path = path.rstrip("/")
    return path if os.path.exists(path) else os.path.join(os.path.dirname(json_path), os.path.basename(path))


def reeval(json_path, mode, n_samples, chunk, n_discard):
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from CouplingModel import ModelFactory

    from Hydra.NN.utils import resolve_best_orbax
    from Hydra.saveNload import load_vstate

    art = json.load(open(json_path))
    folder = resolve_best_orbax(os.path.join(local(art["_artifacts"]["best"], json_path), mode))
    vs = load_vstate(art, params_path=f"{folder}/parameters", sampler_path=f"{folder}/sampler_state")
    vs.chunk_size = chunk or art["SIM"]["sampler"].get("chunk_vstate")
    vs.n_discard_per_chain = n_discard
    vs.n_samples = n_samples
    H = ModelFactory.init(art["CM"]).get_model().cm.build_hamiltonian(
        vs.hilbert, S_operators=art["CM"].get("S_operators", False))
    st = vs.expect(H.to_jax_operator())
    E, var = float(np.real(st.mean)), float(st.variance)
    N = vs.hilbert.size
    return dict(E=E, E_err=float(st.error_of_mean), var=var, vscore=N * var / E**2,
                tau=float(st.tau_corr), R_hat=float(st.R_hat), n_samples=int(vs.n_samples),
                folder=folder, date=time.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsons", nargs="*", help="results JSONs (or use -d/--where)")
    ap.add_argument("-d", action="append", default=[], help="folders to index")
    ap.add_argument("--where", help="aplot selection over the index")
    ap.add_argument("--mode", default="best", help="keeper mode: best, best_energy, best_vscore")
    ap.add_argument("--samples", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=None, help="default: the run's chunk_vstate")
    ap.add_argument("--discard", type=int, default=16, help="sweeps discarded per chain")
    ap.add_argument("--force", action="store_true", help="redo modes already in the sidecar")
    a = ap.parse_args()

    paths = list(a.jsons)
    if a.d or a.where:
        paths += list(select(load_index(a.d or ["."]), a.where)["path"])
    if not paths:
        ap.error("no runs: give JSONs or -d/--where")
    for p in paths:
        side = sidecar(p)
        done = json.load(open(side)) if os.path.exists(side) else {}
        if a.mode in done and not a.force:
            print(f"[reeval] done already: {p}")
            continue
        try:
            st = reeval(p, a.mode, a.samples, a.chunk, a.discard)
        except Exception as e:  # one broken run must not stop the batch
            print(f"[reeval] FAILED {p}: {type(e).__name__}: {e}")
            continue
        done[a.mode] = st
        json.dump(done, open(side, "w"), indent=2)
        print(f"[reeval] {Path(p).name}: E = {st['E']:.6f} ± {st['E_err']:.6f}  "
              f"V = {st['vscore']:.3e}  R_hat = {st['R_hat']:.3f}  ({st['n_samples']} samples)")


if __name__ == "__main__":
    main()
