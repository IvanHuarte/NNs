#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python
"""
aplot: figures and tables from the experiment index (docs/08, section 8.8).

    aplot [xy|trace|table] key=value ... [--where EXPR] [--label TXT]
          [--app key=value ... [--where EXPR] [--label TXT]] ...
          [--for COL[,COL]] [-d DIR ...] [-o OUT] [--name NAME] [--title TXT]
          [--fmt png|pdf|svg] [--refresh] [--show]

    aplot y=E_site_spin x=J2 hue=L --where "net='CvT3' and dtype=='float32' and not vscore>0.1"
    aplot y=E_site_spin x=J2 hue=L --where "irrep=='000'" --app --where "irrep==001"
    aplot y=gap x=irrep hue=L --where "net='ViT2D'" --for dtype
    aplot y=E_site_spin x="1/L**3" fit=lin --where "irrep=='000'"   # finite size
    aplot y=E_site_spin x=var fit=lin hue=L                       # variance extrapolation
    aplot trace y=E_site_spin smooth=50 sched=1 --where "job~='c8_CvT3'"
    aplot table sort=E50 --where "L==10"

Keys of each kind: autoplots/plots/{xy,trace,table}.py. Columns and the
--where syntax: autoplots/index.py.
--app starts another selection drawn on the same axes; it inherits the keys
of the previous one except --where and --label.
--for draws one figure per value (or combination of values) present in the
selections. Every figure is saved with a .csv of the plotted values; both
carry the command and the git version.
From Python: from autoplots.aplot import main; main("y=E x=J2 --where ...").
"""

import os
import re
import shlex
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # trace sched=1 imports jax: never take the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from autoplots.index import ensure, load_index, ordered, select  # noqa: E402
from autoplots.plots import table, trace, xy  # noqa: E402
from autoplots.plots.common import git_version, save  # noqa: E402

KINDS = {"xy": xy, "trace": trace, "table": table}
STYLE = Path(__file__).resolve().parent / "aplot.mplstyle"
VALUED = {"-d", "-o", "--for", "--where", "--label", "--name", "--title", "--fmt"}


def parse(argv):
    """(kind, [spec per selection], global options)."""
    kind = argv[0] if argv and argv[0] in KINDS else "xy"
    argv = argv[1:] if argv and argv[0] in KINDS else argv
    g = dict(dirs=[], out="Figures", fmt="png", name=None, title=None, refresh=False, show=False)
    specs = [{}]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in VALUED:
            if i + 1 >= len(argv):
                raise SystemExit(f"aplot: {tok} needs a value")
            val = argv[i + 1]
            i += 1
            if tok in ("--where", "--label"):
                specs[-1][tok[2:]] = val
            elif tok == "-d":
                g["dirs"].append(val)
            elif tok == "-o":
                g["out"] = val
            elif tok == "--for":
                g["for"] = val.split(",")
            else:
                g[tok[2:]] = val
        elif tok == "--app":
            specs.append({k: v for k, v in specs[-1].items() if k not in ("where", "label")})
        elif tok in ("--refresh", "--show"):
            g[tok[2:]] = True
        elif "=" in tok and not tok.startswith("-"):
            k, v = tok.split("=", 1)
            specs[-1][k] = v
        else:
            raise SystemExit(f"aplot: unknown argument {tok!r} (aplot -h)")
        i += 1
    if kind == "xy" and not all("x" in s and "y" in s for s in specs):
        raise SystemExit("aplot xy: every selection needs x= and y=")
    return kind, specs, g


def default_name(kind, s):
    clean = lambda v: re.sub(r"[^\w.-]+", "", v)
    if kind == "xy":
        return clean(f"{s['y']}_vs_{s['x']}" + (f"_by_{s['hue']}" if s.get("hue") else ""))
    return clean(f"trace_{s.get('y', 'energy')}") if kind == "trace" else "table"


def emit(kind, specs, sels, g, base, title, cmd):
    if not any(len(s) for s in sels):
        print(f"[aplot] {base}: no runs, skipped")
        return
    path = Path(g["out"]) / base
    if kind == "table":
        text, ext = table.run(specs, sels)
        print(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        comment = {"md": "<!-- {} -->", "tex": "% {}", "csv": "# {}"}[ext]
        path.with_suffix(f".{ext}").write_text(comment.format(f"aplot {cmd} | git {git_version()}") + "\n" + text)
        print(f"[aplot] {path}.{ext}")
        return
    import matplotlib.pyplot as plt

    fig, data = KINDS[kind].run(specs, sels)
    if title:
        fig.suptitle(title)
        fig.tight_layout()
    save(fig, data, path, g["fmt"], cmd)
    if g["show"]:
        plt.show()
    plt.close(fig)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    elif isinstance(argv, str):
        argv = shlex.split(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    kind, specs, g = parse(argv)
    if not g["show"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(STYLE)
    df = load_index(g["dirs"] or ["."], refresh=g["refresh"])
    sels = [select(df, s.get("where")) for s in specs]
    for s, sel in zip(specs, sels):
        print(f"[aplot] {len(sel)} runs  <- {s.get('where') or 'all'}")
    cmd = " ".join(shlex.quote(a) for a in argv)
    base = g["name"] or default_name(kind, specs[0])

    keys = g.get("for")
    if not keys:
        emit(kind, specs, sels, g, base, g["title"], cmd)
        return
    sels = [ensure(sel, keys) for sel in sels]
    combos = ordered(tuple(r) for sel in sels for r in sel[keys].itertuples(index=False))
    for combo in combos:
        subs = [sel[np.logical_and.reduce([sel[k].map(lambda x, v=v: x == v) for k, v in zip(keys, combo)])]
                for sel in sels]
        tag = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
        title = ", ".join(f"{k} = {v}" for k, v in zip(keys, combo))
        emit(kind, specs, subs, g, re.sub(r"[^\w.-]+", "", f"{base}_{tag}"),
             f"{g['title']}  ({title})" if g["title"] else title, cmd)


if __name__ == "__main__":
    main()
