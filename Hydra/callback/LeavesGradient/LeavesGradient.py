"""
LeavesGradient: leaf-by-leaf view of the parameters, the SR update and the
log-derivatives, for debugging the training of an arbitrary architecture.

Every `do_each` (global) steps it prints a tree aggregated to `depth`
(module rows, with the positional codes of the schedule), each row with

    nº | |θ| | |dp|/|θ| | |Δθ|/|θ| | |O_re| | |O_im| | dead | ruler

|dp| is the SR direction (before the learning rate) and |Δθ| the step
actually applied (θ_t − θ_{t−1}: learning rate, momentum and schedule masks
included; a copy of the parameters is taken one step before each report).
The ruler places log10(|Δθ|/|θ|) of the row relative to the median of all
leaves (centre │, ±3 decades), so layers that learn much faster or slower
than the rest stand out at a glance. Leaves that are outliers (factor
`outlier` above/below the median, frozen, dead or non-finite) are listed
separately, whatever the depth. Scalars whose derivative is exactly zero
are global constants of log ψ (normalisation / global phase) and are not
reported.

It also keeps the history of log10(|Δθ|/|θ|) per leaf and, at the end of the
run (finalize), saves a heat map leaves × steps to
`<sim>_leaves_heatmap.png` (+ the raw data `<sim>_leaves.npz`): it shows which
layers learn when, layers that freeze or blow up, and the effect of every
stage or schedule period.

config0 "leaves_gradient_setup":
    do_each      steps between reports (default 50)
    depth        depth of the printed tree (default 2; 0 = every leaf)
    jacobian     also the log-derivatives O (default true; costs a Jacobian
                 on `n_jacobian` samples)
    n_jacobian   samples for the Jacobian (default 64)
    outlier      factor that flags a leaf (default 100)
    show         "tree" (default), "outliers" (only the flagged leaves) or "none"
                 (only the heat map at the end)
"""

from __future__ import annotations

import math

import jax
import numpy as np

from ..introspection import (
    aggregate,
    centred_log_derivatives,
    finish_row,
    leaf_table,
    path_codes,
    strip_wrappers,
    subsample,
)

DECADES, HALF = 3.0, 12  # ruler: ±3 decades around the median, 12 chars per side


def _ruler(v, med):
    """'······│···●··': position of v relative to the median (log scale)."""
    if not (isinstance(v, float) and math.isfinite(v)) or v <= 0 or not math.isfinite(med) or med <= 0:
        return " " * (2 * HALF + 1)
    x = math.log10(v / med) / DECADES
    i = HALF + int(round(max(-1.0, min(1.0, x)) * HALF))
    cells = ["·"] * (2 * HALF + 1)
    cells[HALF] = "│"
    cells[i] = "◀" if x < -1 else "▶" if x > 1 else "●"
    return "".join(cells)


def _f(v, fmt="{:.2e}"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return fmt.format(v)


class LeavesGradient:
    def __init__(self, setup, log_prefix=None):
        if "metrics" in setup:  # old "mean|std|max|min" setup
            print("[LeavesGradient] 'metrics' is no longer used; see the module docstring")
        self.do_each = int(setup.get("do_each", 50))
        self.depth = int(setup.get("depth", 2))
        self.jacobian = bool(setup.get("jacobian", True))
        self.n_jacobian = int(setup.get("n_jacobian", 64))
        self.outlier = float(setup.get("outlier", 100.0))
        self.show = setup.get("show", "tree")
        self.log_prefix = log_prefix
        self.step = -1
        self.stage = 0
        self.codes = None
        self.history = []  # (global step, stage, {leaf path: log10 |Δθ|/|θ|})
        self.stage_starts = []
        self.prev = None

    # hooks ---------------------------------------------------------------

    def new_stage(self, stage=None):
        self.stage = self.stage + 1 if stage is None else stage
        self.codes = None
        self.prev = None
        self.stage_starts.append(self.step + 1)

    def finalize(self):
        if self.log_prefix and self.history:
            self.save_heatmap(self.log_prefix + "_leaves_heatmap.png")

    # ---------------------------------------------------------------------

    def __call__(self, step, log_data, driver):
        self.step += 1
        if (self.step + 1) % self.do_each == 0:
            self.prev = jax.device_get(driver.state.parameters)
        if self.step % self.do_each != 0:
            return True
        vstate = driver.state
        params = jax.device_get(vstate.parameters)
        dp = jax.device_get(getattr(driver, "_dp", None))
        prev, self.prev = self.prev, None
        O = None
        if self.jacobian:
            s = subsample(vstate.samples, self.n_jacobian, seed=self.step)
            O = centred_log_derivatives(vstate, s)
        if self.codes is None:
            self.codes = path_codes(params)
        rows = leaf_table(params, dp, O, prev=prev)
        for r in rows:
            finish_row(r, {"ss_dp": sum(x.get("ss_dp", 0.0) for x in rows)})
        self.key = "rel_step" if any("rel_step" in r for r in rows) else "rel"

        def lg(r):  # history: only the applied step (NaN without a previous copy)
            v = r.get("rel_step", math.nan)
            return math.log10(v) if math.isfinite(v) and v > 0 else np.nan

        self.history.append((self.step, self.stage,
                             {"/".join(strip_wrappers(r["path"])): lg(r) for r in rows}))
        if self.show != "none":
            self.display(rows)
        return True

    # ---------------------------------------------------------------------

    def _outliers(self, rows):
        k = self.key
        vals = sorted(r[k] for r in rows if math.isfinite(r.get(k, math.nan)) and r[k] > 0)
        med = vals[len(vals) // 2] if vals else math.nan
        out = []
        for r in rows:
            if r.get("gauge"):
                continue
            why = []
            if not r["finite"]:
                why.append("NaN/Inf")
            v = r.get(k, math.nan)
            if math.isfinite(v) and math.isfinite(med):
                if v > self.outlier * med:
                    why.append(f"{v / med:.0f}× la mediana")
                elif 0 < v < med / self.outlier:
                    why.append(f"1/{med / v:.0f} de la mediana")
                elif v == 0:
                    why.append("congelada")
            if r.get("n_dead", 0) > 0.5 * r["size"]:
                why.append(f"{100 * r['n_dead'] / r['size']:.0f}% muertos")
            if why:
                out.append((r, ", ".join(why)))
        return out, med

    def display(self, rows):
        from rich import box
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table

        console = Console()
        cols = [("nº", "size", "{:d}"), ("|θ|", "theta", "{:.2e}"), ("|dp|/|θ|", "rel", "{:.2e}")]
        if self.key == "rel_step":
            cols.append(("|Δθ|/|θ|", "rel_step", "{:.2e}"))
        if self.jacobian:
            cols += [("|O_re|", "o_re", "{:.2e}"), ("|O_im|", "o_im", "{:.2e}"), ("muertos", "dead", "{:.0%}")]
        outliers, med = self._outliers(rows)
        name = "|Δθ|/|θ|" if self.key == "rel_step" else "|dp|/|θ|"

        def table(title):
            t = Table(title=title, box=box.SIMPLE_HEAD, header_style="bold cyan", title_justify="left")
            t.add_column("módulo / hoja")
            for h, _, _ in cols:
                t.add_column(h, justify="right")
            t.add_column(f"{name} vs mediana (±{DECADES:.0f} déc.)")
            return t

        if self.show == "tree":
            depth = self.depth if self.depth > 0 else 99
            mods, total = aggregate(rows, depth, self.codes)
            t = table(f"LeavesGradient · step {self.step} · stage {self.stage} · mediana {name} {_f(med)}")
            for m in mods + [dict(total, label="TOTAL")]:
                t.add_row(escape(m["label"]), *(_f(m.get(k), f) for _, k, f in cols), _ruler(m.get(self.key), med),
                          style="bold" if m.get("label") == "TOTAL" else None)
            console.print(t)
        if outliers:
            t = table(f"Hojas atípicas (factor {self.outlier:g})")
            t.add_column("motivo", style="yellow")
            for r, why in outliers:
                t.add_row(escape("/".join(strip_wrappers(r["path"]))), *(_f(r.get(k), f) for _, k, f in cols),
                          _ruler(r.get(self.key), med), why)
            console.print(t)
        elif self.show == "outliers":
            console.print(f"[green]LeavesGradient step {self.step}: ninguna hoja atípica[/]")

    def save_heatmap(self, path):
        import matplotlib.pyplot as plt

        leaves = []
        for _, _, d in self.history:
            for k in d:
                if k not in leaves:
                    leaves.append(k)
        steps = [h[0] for h in self.history]
        data = np.full((len(leaves), len(steps)), np.nan)
        for j, (_, _, d) in enumerate(self.history):
            for i, k in enumerate(leaves):
                data[i, j] = d.get(k, np.nan)
        np.savez(path.replace("_heatmap.png", ".npz"), steps=np.array(steps), leaves=np.array(leaves), log10_rel=data)

        h = max(4.0, 0.18 * len(leaves) + 1.5)
        fig, ax = plt.subplots(figsize=(12, min(h, 60)))
        im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest",
                       extent=(-0.5, len(steps) - 0.5, len(leaves) - 0.5, -0.5))
        def short(k):
            p = k.split("/")
            return k if len(p) <= 3 else p[0] + "/…/" + "/".join(p[-2:])

        ax.set_yticks(range(len(leaves)))
        ax.set_yticklabels([short(k) for k in leaves], fontsize=6)
        xt = np.linspace(0, len(steps) - 1, min(10, len(steps))).astype(int)
        ax.set_xticks(xt)
        ax.set_xticklabels([steps[i] for i in xt])
        ax.set_xlabel("step")
        for s in self.stage_starts:
            j = np.searchsorted(steps, s)
            ax.axvline(j - 0.5, color="w", lw=1)
        fig.colorbar(im, ax=ax, label="log10 |Δθ|/|θ|")
        ax.set_title("Paso relativo por hoja (blanco: cambio de stage)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[LeavesGradient] heat map -> {path}")
