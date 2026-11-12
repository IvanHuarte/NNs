"""
Table of the selected runs (several selections: first column `set`).

keys  cols=a,b,c          columns or expressions (default: DEFAULT, missing ones skipped)
      sort=COL            '-COL' descending; NaN last
      fmt=md|tex|csv
      top=N               first N rows of each selection
"""

import numpy as np
import pandas as pd

from ..index import column

DEFAULT = "job,name,L,irrep,dtype,E,E_site_spin,E50,vscore,epochs,s_it,time_h,status"


def cell(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.6g}"
    return "" if v is None else str(v)


def markdown(t):
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def run(specs, sels):
    s0 = specs[0]
    asked = "cols" in s0
    cols = s0.get("cols", DEFAULT).split(",")
    parts = []
    for i, (spec, sel) in enumerate(zip(specs, sels)):
        use = [c for c in cols if asked or c in sel]
        t = pd.DataFrame({c: column(sel, c) for c in use})
        if spec.get("sort"):
            key = spec["sort"]
            t = t.assign(_k=column(sel, key.lstrip("-"))).sort_values(
                "_k", ascending=not key.startswith("-"), na_position="last").drop(columns="_k")
        if spec.get("top"):
            t = t.head(int(spec["top"]))
        if len(specs) > 1:
            t.insert(0, "set", spec.get("label") or spec.get("where") or f"set {i}")
        parts.append(t)
    t = pd.concat(parts)
    fmt = s0.get("fmt", "md")
    if fmt == "csv":
        return t.to_csv(index=False), "csv"
    if fmt == "tex":
        return t.to_latex(index=False, float_format="%.6g", na_rep=""), "tex"
    return markdown(t), "md"
