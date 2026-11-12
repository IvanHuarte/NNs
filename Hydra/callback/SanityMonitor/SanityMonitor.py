"""
SanityMonitor: health checks of a running VMC simulation, for any
architecture. Every `do_each` (global) steps it computes the selected
metrics, diagnoses them (ok / info / warn / error) and prints them; with a
log path it also appends every check to `<sim>_sanity.jsonl`.

config0 "sanity_setup":
    do_each       steps between checks (default 50)
    mode          "rich" (default) or "simple" (plain text, for log files)
    verbose       0-4 (see display_diagnosis); default 2 (warnings + errors)
    metrics       {name: bool} among energy, sampling, optimization,
                  wavefunction, gradients, local_energy, magnetization
                  (old names: phase -> wavefunction, sample_histogram -> magnetization)
    module_depth  depth of the per-module tables (default 1)
    n_jacobian    samples used for the log-derivatives / NTK (default 128)
    log           write <sim>_sanity.jsonl (default true)
"""

from __future__ import annotations

import json
import math

import jax
import numpy as np

from ..introspection import path_codes
from .diagnose import diagnose_metrics
from .display_diagnosis import display_diagnosis_sanity_monitor, display_diagnosis_simple
from .metrics import ALIASES, METRICS, calc_metrics

DEFAULT_METRICS = ("energy", "sampling", "optimization", "wavefunction")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class SanityMonitor:
    def __init__(self, sanity_setup, N=None, E_ref=None, log_path=None):
        self.do_each = int(sanity_setup.get("do_each", 50))
        self.verbose = int(sanity_setup.get("verbose", 2))
        self.mode = sanity_setup.get("mode", "rich")
        if self.mode not in ("rich", "simple"):
            raise ValueError(f"sanity_setup.mode={self.mode!r}: use 'rich' or 'simple'")
        sel = sanity_setup.get("metrics", {m: True for m in DEFAULT_METRICS})
        names = []
        for k, on in sel.items():
            k = ALIASES.get(k, k)
            if k not in METRICS:
                raise ValueError(f"sanity metric {k!r} unknown; use some of {list(METRICS)}")
            if on and k not in names:
                names.append(k)
        self.selection = names
        self.ctx = {
            "N": N,
            "E_ref": E_ref,
            "depth": int(sanity_setup.get("module_depth", 1)),
            "n_jacobian": int(sanity_setup.get("n_jacobian", 128)),
            "history": [],
            "codes": None,
            "step": 0,
        }
        self.log_path = log_path if sanity_setup.get("log", True) else None
        self.step = -1

    # hooks used by VMC_simulation.train
    def new_stage(self, stage=None):
        self.ctx["codes"] = None  # new architecture: recompute the codes
        self.ctx["history"].clear()
        self.ctx["prev_params"] = None

    def __call__(self, step, log_data, driver):
        self.step += 1
        # parameters of the step before the check: the next call sees the
        # step actually applied (lr, momentum and masks included). Taken
        # aside so that with do_each 1 this check still uses the previous copy
        nxt = None
        if (self.step + 1) % self.do_each == 0 and "optimization" in self.selection:
            nxt = jax.device_get(driver.state.parameters)
        if self.step == 0 or self.step % self.do_each != 0:
            self.ctx["prev_params"] = nxt
            return True
        if self.ctx["N"] is None:
            self.ctx["N"] = driver.state.hilbert.size
        if self.ctx["codes"] is None:
            self.ctx["codes"] = path_codes(jax.device_get(driver.state.parameters))
        self.ctx["step"] = self.step

        metrics = calc_metrics(driver, log_data, self.ctx, self.selection)
        self.ctx["prev_params"] = nxt
        diagnosis = diagnose_metrics(metrics)
        self.ctx["history"].append(metrics)
        if len(self.ctx["history"]) > 2:
            self.ctx["history"].pop(0)

        if self.mode == "rich":
            display_diagnosis_sanity_monitor(diagnosis, self.verbose, step=self.step)
        else:
            display_diagnosis_simple(diagnosis, self.verbose, step=self.step)
        if self.log_path:
            rec = {"step": self.step, "metrics": metrics,
                   "status": {c: [(x["name"], x["status"]) for x in d["checks"]]
                              for c, d in diagnosis.items() if c != "overall"},
                   "overall": diagnosis["overall"]["status"]}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(_jsonable(rec)) + "\n")
        return True
