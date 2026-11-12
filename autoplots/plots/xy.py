"""
y against x for one or more selections (--app): one curve per hue value, one
panel per col value. Colour = hue value, line style = selection.

keys  y x                 columns or expressions ('1/L**3', 'E_re/N')
      hue col             one curve / one panel per value
      agg=best|mean|median|all
                          several runs on the same point: the one with the
                          lowest `best` column (default), mean/median with std
                          error bars, or every run as a scatter
      best=E              column that ranks the runs for agg=best ('-col': highest)
      yerr=COL ref=COL    error bars / a reference curve (dashed) from a column
      fit=lin|quad        polynomial fit per curve, extrapolated to x=0
      logx=1 logy=1 xlim=a,b ylim=a,b
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..index import column, ordered
from .common import label

STYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v"]


def frame(sel, spec):
    """x, y and the grouping columns of a selection; rows without x or y dropped."""
    out = pd.DataFrame(index=sel.index)
    out["x"] = column(sel, spec["x"])
    out["y"] = column(sel, spec["y"])
    for k in ("hue", "col"):
        if spec.get(k):
            out[k] = column(sel, spec[k]).astype(object).where(lambda s: s.notna(), "?")
    for k in ("ref", "yerr"):
        if spec.get(k):
            out[k] = column(sel, spec[k])
    best = spec.get("best", "E")
    sign = -1 if best.startswith("-") else 1
    out["best"] = (sign * column(sel, best.lstrip("-")).astype(float)).fillna(np.inf)
    out["path"] = sel["path"]
    dropped = out[["x", "y"]].isna().any(axis=1)
    if dropped.any():
        print(f"[xy] {dropped.sum()} runs without {spec['x']} or {spec['y']} left out")
    return out[~dropped]


def aggregate(g, agg):
    """One point per x value."""
    if agg == "all":
        return g.assign(n=1)
    n = g.groupby("x", sort=False).size()
    if agg in ("mean", "median"):
        cols = {"y": ("y", agg), "yerr": ("y", "std")}
        if "ref" in g:
            cols["ref"] = ("ref", "mean")
        return g.groupby("x", sort=False).agg(**cols).reset_index().assign(n=lambda p: n.loc[p.x].values)
    if agg != "best":
        raise ValueError(f"agg={agg}: use best, mean, median or all")
    p = g.loc[g.groupby("x", sort=False)["best"].idxmin()]
    return p.assign(n=n.loc[p.x].values)


def fit(ax, p, kind, color, name):
    """Polynomial fit of the curve, drawn from x=0; returns (intercept, error)."""
    deg = {"lin": 1, "quad": 2}[kind]
    x, y = p["x"].to_numpy(float), p["y"].to_numpy(float)
    if len(x) < deg + 1:
        print(f"[fit] {name}: {len(x)} points, not enough for fit={kind}")
        return None
    w = 1 / p["yerr"].to_numpy(float) if "yerr" in p and (p["yerr"] > 0).all() else None
    if len(x) > deg + 2:
        c, cov = np.polyfit(x, y, deg, w=w, cov=True)
        err = float(np.sqrt(cov[-1, -1]))
    else:
        c, err = np.polyfit(x, y, deg, w=w), np.nan
    xs = np.linspace(0, x.max(), 200)
    ax.plot(xs, np.polyval(c, xs), color=color, lw=1, ls="--", alpha=0.7,
            label=f"fit: {c[-1]:.6g} ± {err:.2g}")
    print(f"[fit] {name}: y(x=0) = {c[-1]:.10g} ± {err:.3g}  ({len(x)} points, {kind})")
    return c[-1], err


def run(specs, sels):
    frames = [frame(sel, spec) for sel, spec in zip(sels, specs)]
    filled = [f for f in frames if len(f)]
    categorical = not all(pd.api.types.is_numeric_dtype(f["x"]) for f in filled)
    cats = ordered(pd.concat([f["x"] for f in filled])) if categorical else None
    cvals = ordered(pd.concat([f["col"] for f in filled if "col" in f])) if any("col" in f for f in filled) else [None]

    fig, axes = plt.subplots(1, len(cvals), squeeze=False, sharey=True,
                             figsize=(5.5 * len(cvals) + 1, 4.2))
    axes = axes[0]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors, rows = {}, []
    multi = len(specs) > 1

    for i, (spec, f) in enumerate(zip(specs, frames)):
        tag = spec.get("label") or spec.get("where") or f"set {i}"
        agg = spec.get("agg", "best")
        hvals = ordered(f["hue"]) if "hue" in f else [None]
        for cval in cvals:
            for hval in hvals:
                m = np.ones(len(f), bool)
                if cval is not None:
                    m &= (f["col"] == cval).to_numpy()
                if hval is not None:
                    m &= (f["hue"] == hval).to_numpy()
                if not m.any():
                    continue
                p = aggregate(f[m], agg)
                p["xpos"] = p["x"].map(cats.index) if categorical else p["x"]
                p = p.sort_values("xpos")
                name = " ".join(s for s in (tag if multi else "",
                                            f"{spec['hue']}={hval}" if hval is not None else "") if s) or spec["y"]
                color = colors.setdefault(hval if hval is not None else ("set", i), cycle[len(colors) % len(cycle)])
                ax = axes[cvals.index(cval)]
                kw = dict(color=color, ls="none" if agg == "all" else STYLES[i % 4],
                          marker=MARKERS[i % 5], label=name)
                if "yerr" in p and p["yerr"].notna().any():
                    ax.errorbar(p["xpos"], p["y"], yerr=p["yerr"], capsize=2, **kw)
                else:
                    ax.plot(p["xpos"], p["y"], **kw)
                if "ref" in p:
                    ax.plot(p["xpos"], p["ref"], color=color, ls=(0, (1, 1)), marker="x", lw=1,
                            label=f"{name} {spec['ref']}")
                out = p.drop(columns=["xpos", "best"]).assign(curve=name, col=cval)
                if spec.get("fit") and not categorical:
                    res = fit(ax, p, spec["fit"], color, name)
                    if res:
                        out = pd.concat([out, pd.DataFrame([dict(curve=f"{name} fit x=0", col=cval,
                                                                 x=0.0, y=res[0], yerr=res[1])])])
                rows.append(out)

    s0 = specs[0]
    for ax, cval in zip(axes, cvals):
        if cval is not None:
            ax.set_title(f"{s0['col']} = {cval}")
        ax.set_xlabel(label(s0["x"]))
        if categorical:
            ax.set_xticks(range(len(cats)), [str(c) for c in cats], rotation=45 if len(cats) > 6 else 0)
        if s0.get("logx"):
            ax.set_xscale("log")
        if s0.get("logy"):
            ax.set_yscale("log")
        if s0.get("xlim"):
            ax.set_xlim(*map(float, s0["xlim"].split(",")))
        if s0.get("ylim"):
            ax.set_ylim(*map(float, s0["ylim"].split(",")))
        if ax.get_legend_handles_labels()[0] and len(ax.get_legend_handles_labels()[0]) <= 20:
            ax.legend()
    axes[0].set_ylabel(" / ".join(dict.fromkeys(label(s["y"]) for s in specs)))
    fig.tight_layout()
    data = pd.concat(rows) if rows else None
    return fig, data
