"""
Index of experiments: one row per results JSON, one column per attribute.

    from autoplots.index import load_index, select
    df = load_index(["Simulations_claude_cesga"])
    sel = select(df, "net='CvT3' and dtype=='float32' and not vscore>0.1")

Columns (all optional: old JSON schemas give NaN where a field is missing)

    path job host model size L N system      where and what system
    <every CM param>  (J1, J2, fields...)    any model, also future ones
    <every scalar in results>                E_best, vscore, time_exe, fidelity_irrep...
    name net head irrep kx ky z2 dtype seed nsamples epochs
    E units E_spin E_site E_site_spin var    E = results.E_best; *_spin in S = sigma/2 units
    E50 vscore50 nan_epoch stages trace      keeper trace: best 50-epoch mean of the last stage
    E_ref err_rel                            ED of the sector, else refs.json
    E_re E_re_err vscore_re ...              reeval.py sidecar (<json>_reeval.json)
    time_h s_it status

Any other name used in a query or a plot is searched in the JSONs with
utils.find_value (first match, 'a/b/c' paths allowed) and added as a column.
`gap` (E_spin minus the lowest E_spin of the same system) is computed on the
current selection.

The index is cached in autoplots/.cache/index.pkl; a run is re-read only when
its JSON, trace or sidecar change. Nothing here imports Hydra (jax would take
the GPU).
"""

import glob
import json
import os
import pickle
import re
import warnings
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import find_value, natural_key

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "index.pkl"
REFS = HERE / "refs.json"
CACHE_VERSION = 2  # bump when row() or trace_stats() change

# modules that route or symmetrise others (Hydra/label_utils.FLUX_MODULES + wrappers)
WRAPPERS = {"NNWrapper", "Z2", "Traslations", "Sequential", "Sequivariant",
            "SingleModule", "SplitTraining", "Transversal"}
HEAD_ABBR = {"OutputHead": "OH", "SiteHead": "SH", "DeepOutputHead": "DOH",
             "ComplexHead": "CH", "SzaboOutput": "SzO"}
SKIP_DIRS = {".venv", ".git", "__pycache__", ".cache"}
TRACE_WINDOW = 50


# ---------------------------------------------------------------- discovery


def find_results(dirs):
    """Every results JSON under dirs (orbax folders are not walked)."""
    out = []
    for d in dirs:
        for root, subdirs, files in os.walk(d):
            subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS
                          and not s.endswith(".orbax") and not s.startswith("pytest-of")]
            out += [os.path.join(root, f) for f in files
                    if "results" in f and f.endswith(".json") and not f.endswith("_reeval.json")]
    return sorted(os.path.abspath(p) for p in out)


def sidecar(path):
    return path[: -len(".json")] + "_reeval.json"


def trace_of(path):
    """Keeper trace of the run (same folder as the JSON), newest if several."""
    traces = glob.glob(os.path.join(os.path.dirname(path), "*_trace.txt"))
    return max(traces, key=os.path.getmtime) if traces else None


def mtime(p):
    return os.path.getmtime(p) if p and os.path.exists(p) else None


# ---------------------------------------------------------------- one run


def scalar(v):
    if isinstance(v, list) and all(isinstance(x, (int, float, str, bool)) for x in v):
        return tuple(v)
    return v if isinstance(v, (int, float, str, bool)) or v is None else None


def modules(tree):
    """'module' values of a Hydra NN setup in depth-first order."""
    out = []
    if isinstance(tree, dict):
        if isinstance(tree.get("module"), str):
            out.append(tree["module"])
        for v in tree.values():
            out += modules(v)
    return out


def irrep_code(irrep, z2):
    """Same digits as Hydra.label_utils.irrep_code: (4, 4), 1 -> '441'; (10, 0), 1 -> '10.0.1'."""
    vals = [int(v) for v in irrep] + [int(z2)]
    return ("." if any(v >= 10 for v in vals) else "").join(map(str, vals))


def trace_stats(trace):
    """Best rolling mean of the energy in the last stage, first NaN step, number of stages."""
    out = dict(E50=np.nan, vscore50=np.nan, nan_epoch=np.nan, stages=np.nan, trace_epochs=np.nan)
    if not trace:
        return out
    try:
        d = np.atleast_2d(np.loadtxt(trace))  # step stage local_step energy variance vscore
    except ValueError:
        return out
    if d.size == 0 or d.shape[1] < 6:
        return out
    bad = ~np.isfinite(d[:, 3])
    out["nan_epoch"] = d[bad, 0][0] if bad.any() else np.nan
    out["stages"] = len(np.unique(d[:, 1]))
    out["trace_epochs"] = len(d)
    last = d[(d[:, 1] == d[-1, 1]) & ~bad & (d[:, 4] >= 0)]
    if len(last):  # a diverging run samples energies far below the physical ones: drop them
        med = np.median(last[:, 3])
        mad = np.median(np.abs(last[:, 3] - med))
        last = last[last[:, 3] >= med - 50 * mad]
    if len(last):
        w = min(TRACE_WINDOW, max(1, len(last) // 2))
        kernel = np.ones(w) / w
        E = np.convolve(last[:, 3], kernel, mode="valid")
        V = np.convolve(last[:, 5], kernel, mode="valid")
        i = int(np.argmin(E))
        out["E50"], out["vscore50"] = E[i], V[i]
    return out


def row(path):
    a = json.load(open(path))
    res = a.get("results") or {}
    cm = a.get("CM") or {}
    sim = a.get("SIM") or {}
    nn = a.get("NN") or {}
    meta = a.get("metadata") or {}
    params = cm.get("params") or a.get("coupling_model") or {}
    size = cm.get("size") or (a.get("lattice") or {}).get("size")
    model = cm.get("name") or "?"

    r = {k: scalar(v) for k, v in res.items() if scalar(v) is not None}
    r.update({k: scalar(v) for k, v in params.items() if scalar(v) is not None})

    parts = Path(path).parts
    r["path"] = path
    r["job"] = parts[parts.index(model) - 1] if model in parts else Path(path).parent.name
    r["host"] = next((h for h in ("cesga", "garmonegro") if h in path), "local")
    r["model"] = model
    if size:
        r["size"] = "x".join(map(str, size))
        r["L"] = size[0]
        r["N"] = int(np.prod(size))
    r["system"] = f"{model} {r.get('size')} " + " ".join(f"{k}={v}" for k, v in sorted(params.items()))

    # network
    r["name"] = re.sub(r"_stage\d+$", "", str(nn.get("name") or (a.get("model_NN") or {}).get("name", "")))
    mods = [m for m in modules(nn.get("setup")) if m not in WRAPPERS]
    r["net"] = mods[0] if mods else None
    r["head"] = HEAD_ABBR.get(mods[1], mods[1]) if len(mods) > 1 else None
    try:
        irrep = find_value(nn, "irrep")
    except KeyError:
        irrep = None
    try:
        z2 = 0 if find_value(nn, "trivial_Z2") else 1
    except KeyError:
        z2 = 0
    if isinstance(irrep, (list, tuple)) and len(irrep) == 2:
        r["kx"], r["ky"], r["z2"] = irrep[0], irrep[1], z2
        r["irrep"] = irrep_code(irrep, z2)
    r["dtype"] = sim.get("network_dtype", "float64" if sim else None)
    r["seed"] = sim.get("seed")
    r["nsamples"] = (sim.get("sampler") or {}).get("nsamples")
    r["epochs"] = (sim.get("schedule") or {}).get("total_epochs")
    r["date"] = meta.get("date")
    r["uuid"] = meta.get("uuid")

    # energies. shortcut: E_spin = E/4 holds for two-spin terms only (fields scale by 1/2)
    E = res.get("E_best")
    r["E"] = E
    r["units"] = res.get("energy_units") or ("S" if cm.get("S_operators") else "Pauli")
    scale = 0.25 if r["units"] == "Pauli" else 1.0
    if E is not None and size:
        r["E_spin"] = E * scale
        r["E_site"] = E / r["N"]
        r["E_site_spin"] = E * scale / r["N"]
        if res.get("vscore") is not None:
            r["var"] = res["vscore"] * E**2 / r["N"]  # vscore = N Var / E^2
    if res.get("E_gr_irrep") is not None:
        r["E_ref"] = res["E_gr_irrep"]
    if res.get("time_exe") is not None:
        r["time_h"] = res["time_exe"] / 3600
        if r["epochs"]:
            r["s_it"] = res["time_exe"] / r["epochs"]
    mssg = str(res.get("mssg", ""))
    if E is None or not np.isfinite(E):
        r["status"] = "nan"
    elif "diverged" in mssg:
        r["status"] = "diverged"  # E is the keeper best before the divergence
    else:
        r["status"] = "ok" if ("success" in mssg or not mssg) else mssg

    r["trace"] = trace_of(path)
    r.update(trace_stats(r["trace"]))

    side = sidecar(path)
    if os.path.exists(side):
        modes = json.load(open(side))
        primary = (res.get("keeper") or {}).get("primary")
        st = modes.get(primary) or next(iter(modes.values()))
        r.update({f"{k}_re" if k != "E_err" else "E_re_err": v for k, v in st.items() if scalar(v) is not None})
    return r


# ---------------------------------------------------------------- index


def load_index(dirs=(".",), refresh=False, verbose=True):
    """DataFrame of every run under dirs (cached, see module docstring)."""
    paths = find_results(dirs)
    cache = {}
    if CACHE.exists() and not refresh:
        try:
            stored = pickle.load(open(CACHE, "rb"))
            cache = stored["rows"] if stored.get("version") == CACHE_VERSION else {}
        except Exception:
            cache = {}
    rows, new = [], 0
    for p in paths:
        sig = (mtime(p), mtime(trace_of(p)), mtime(sidecar(p)))
        hit = cache.get(p)
        if hit is None or hit[0] != sig:
            try:
                hit = (sig, row(p))
            except Exception as e:  # unreadable JSON: report and go on
                print(f"[index] skipped {p}: {type(e).__name__}: {e}")
                continue
            cache[p] = hit
            new += 1
        rows.append(hit[1])
    if new:
        CACHE.parent.mkdir(exist_ok=True)
        pickle.dump({"version": CACHE_VERSION, "rows": cache}, open(CACHE, "wb"))
    if verbose:
        print(f"[index] {len(rows)} runs ({new} read, {len(rows) - new} cached)")
    df = pd.DataFrame(rows)
    return add_refs(df) if len(df) else df


def add_refs(df):
    """
    E_ref from refs.json where the run has no ED; err_rel = (E - E_ref) / |E_ref|.
    refs.json: {model: [{<column>: value, ..., "E_site_spin": e, "source": "..."}]},
    every other key of an entry must match the run (e.g. L, J2, irrep).
    """
    if "E_ref" not in df:
        df["E_ref"] = np.nan
    if REFS.exists():
        for model, entries in json.load(open(REFS)).items():
            for ref in entries:
                keys = {k: v for k, v in ref.items() if k not in ("E_site_spin", "source")}
                if not all(k in df for k in keys):
                    continue
                m = (df["model"] == model) & df["E_ref"].isna()
                for k, v in keys.items():
                    m &= df[k].astype(str) == str(v)
                scale = np.where(df.loc[m, "units"] == "Pauli", 0.25, 1.0)
                df.loc[m, "E_ref"] = ref["E_site_spin"] * df.loc[m, "N"] / scale
    df["err_rel"] = (df["E"] - df["E_ref"]) / df["E_ref"].abs()
    return df


# ---------------------------------------------------------------- columns


KEYWORDS = {"and", "or", "not", "in", "is", "True", "False", "None", "nan", "inf"}


def names_in(expr):
    """Column names used in an expression (quoted strings, methods and calls left out)."""
    expr = re.sub(r"'[^']*'|\"[^\"]*\"", "''", expr)
    names = re.findall(r"`([^`]+)`", expr)
    expr = re.sub(r"`[^`]+`", "_", expr)
    names += [n for n in re.findall(r"(?<![\w.])([A-Za-z_]\w*)(?!\s*\()", expr) if n not in KEYWORDS]
    return list(dict.fromkeys(names))


def ensure(df, names):
    """Add the missing columns: gap, or a value searched in each run's JSON."""
    for name in names:
        if name in df or not len(df):
            continue
        if name == "gap":
            df["gap"] = df["E_spin"] - df.groupby("system")["E_spin"].transform("min")
            continue
        vals, found = [], 0
        for p in df["path"]:
            try:
                vals.append(scalar(find_value(json.load(open(p)), name)))
                found += 1
            except (KeyError, OSError):
                vals.append(None)
        if not found:
            close = get_close_matches(name, [str(c) for c in df.columns], n=5)
            raise KeyError(f"'{name}' is not a column nor a key of the JSONs. Close: {close}")
        df[name] = vals
        print(f"[index] '{name}' read from the JSONs ({found}/{len(df)} runs)")
    return df


def column(df, expr):
    """A column, or an expression of columns (pandas eval): 'E_site_spin', '1/L**3'."""
    ensure(df, [expr] if expr in df or not re.search(r"[^\w/]", expr) else names_in(expr))
    return df[expr] if expr in df else df.eval(expr, engine="python")


def parse_where(expr, df):
    """
    Query sugar on top of pandas query:
        a=b         -> a==b
        a~='re'     -> a.str.contains('re')     (regular expression)
        irrep==001  -> irrep=='001'             (bare number vs a text column)
    and/or/not, <, in [...], `a/b` (JSON path) work as in pandas.
    """
    names = names_in(expr)
    ensure(df, names)
    expr = re.sub(r"(?<![<>=!~])=(?!=)", "==", expr)
    expr = re.sub(r"([\w`]+)\s*~=\s*('[^']*'|\"[^\"]*\")", r"\1.astype('str').str.contains(\2, na=False)", expr)
    for col in names:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):  # text column (object or str)
            expr = re.sub(rf"(?<![\w.])({re.escape(col)})\s*(==|!=)\s*(\d[\w.]*)", r"\1\2'\3'", expr)
    return expr


def select(df, where=None):
    """Rows of df matching where (see parse_where); a copy."""
    if not where:
        return df.copy()
    expr = parse_where(where, df)
    try:
        with warnings.catch_warnings():  # str.contains warns on regex groups: they are fine here
            warnings.simplefilter("ignore", UserWarning)
            return df.query(expr, engine="python").copy()
    except Exception as e:
        raise ValueError(f"--where {where!r} -> {expr!r}: {type(e).__name__}: {e}") from None


def ordered(values):
    """Unique values in natural order (8 < 10, '441' < '10.0.1' by digits)."""
    return sorted(pd.unique(pd.Series(list(values)).dropna()), key=natural_key)
