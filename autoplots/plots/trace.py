"""
Training traces (keeper <sim>_trace.txt) of the selected runs, one line per run.

keys  y=energy            trace column or expression of: step stage local_step
                          energy variance vscore E_site E_site_spin
      x=step|local_step
      hue=job             index column(s) that label each run (a,b)
      smooth=W            rolling mean over W epochs (raw trace drawn faint)
      skip=K              drop the first K epochs
      stage=last          only the last Hydra stage
      sched=1             lr (solid) and diag_shift (dashed) on a second axis
      logy=1 xlim=a,b ylim=a,b
Stage changes: dotted vertical lines. Star: keeper best_step. Without ylim
the y range covers the trace after its first 10 %.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import label

COLS = ["step", "stage", "local_step", "energy", "variance", "vscore"]
STYLES = ["-", "--", ":", "-."]


def read(run):
    d = np.atleast_2d(np.loadtxt(run["trace"]))
    t = pd.DataFrame(d[:, :6], columns=COLS)
    scale = 0.25 if run.get("units") == "Pauli" else 1.0
    t["E_site"] = t["energy"] / run["N"]
    t["E_site_spin"] = t["energy"] * scale / run["N"]
    return t


def schedule_curves(path):
    """lr and diag_shift per epoch, rebuilt from the run's schedule (mode 'A' only)."""
    from Hydra.schedule.schedule import Schedule  # imports jax: aplot keeps it on the CPU

    s = Schedule(json.load(open(path))["SIM"]["schedule"], {"a": ("x",)})  # 'A' -> one main branch
    lr = np.concatenate([np.asarray(arr[0]) for arr, _ in s.schedule(return_array=True)])
    funcs, _ = s.get_diag_shift_schedule()
    ds = np.concatenate([np.asarray(f(np.arange(n))) for f, n in zip(funcs, s.flat_setup()["epochs"])])
    return lr, ds


def run(specs, sels):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = None
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rows, lo, hi, k = [], np.inf, -np.inf, 0
    s0 = specs[0]
    xcol = s0.get("x", "step")

    for i, (spec, sel) in enumerate(zip(specs, sels)):
        y = spec.get("y", "energy")
        hue = spec.get("hue", "job").split(",")
        sel = sel[sel["trace"].notna()] if "trace" in sel else sel.iloc[:0]
        for _, r in sel.iterrows():
            t = read(r)
            if spec.get("stage") == "last":
                t = t[t["stage"] == t["stage"].iloc[-1]]
            t = t[t["step"] >= float(spec.get("skip", 0))]
            if t.empty:
                continue
            t["y"] = t[y] if y in t else t.eval(y, engine="python")
            name = " ".join(str(r.get(h)) for h in hue)
            if not np.isnan(r.get("nan_epoch", np.nan)):
                name += f" (NaN @ {int(r['nan_epoch'])})"
            color = cycle[k % len(cycle)]
            k += 1
            ls = STYLES[i % 4]
            w = int(spec.get("smooth", 0))
            if w > 1:
                t["y_smooth"] = t["y"].rolling(w, min_periods=1).mean()
                ax.plot(t[xcol], t["y"], color=color, lw=0.6, alpha=0.25)
                ax.plot(t[xcol], t["y_smooth"], color=color, ls=ls, label=name)
                shown = t["y_smooth"]
            else:
                ax.plot(t[xcol], t["y"], color=color, ls=ls, lw=1, label=name)
                shown = t["y"]
            tail = shown[t["step"] >= t["step"].iloc[0] + 0.1 * (t["step"].iloc[-1] - t["step"].iloc[0])]
            tail = tail[np.isfinite(tail)]
            if len(tail):
                lo, hi = min(lo, tail.min()), max(hi, tail.max())
            for s in t["step"][t["stage"].diff() > 0] if xcol == "step" else []:
                ax.axvline(s, color=color, ls=":", lw=0.8)
            best = r.get("best_step")
            if best is not None and not np.isnan(best) and xcol == "step":
                hit = t[t["step"] == best]
                if len(hit):
                    ax.plot(best, shown[hit.index[0]], marker="*", ms=12, color=color, mec="k", ls="none")
            if spec.get("sched"):
                try:
                    lr, ds = schedule_curves(r["path"])
                    ax2 = ax2 or ax.twinx()
                    ax2.plot(np.arange(len(lr)), lr, color=color, lw=0.8, alpha=0.6)
                    ax2.plot(np.arange(len(ds)), ds, color=color, lw=0.8, alpha=0.6, ls="--")
                except Exception as e:
                    print(f"[trace] no schedule for {name}: {type(e).__name__}: {e}")
            rows.append(t.assign(run=name, path=r["path"]))

    ax.set_xlabel(label(xcol))
    ax.set_ylabel(" / ".join(dict.fromkeys(label(s.get("y", "energy")) for s in specs)))
    if s0.get("logy"):
        ax.set_yscale("log")
    if s0.get("xlim"):
        ax.set_xlim(*map(float, s0["xlim"].split(",")))
    if s0.get("ylim"):
        ax.set_ylim(*map(float, s0["ylim"].split(",")))
    elif np.isfinite(lo) and hi > lo:
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    if ax2 is not None:
        ax2.set_yscale("log")
        ax2.set_ylabel("lr (—), diag_shift (--)")
        ax2.grid(False)
    if 0 < k <= 15:
        ax.legend()
    fig.tight_layout()
    return fig, pd.concat(rows) if rows else None
