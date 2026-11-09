"""
Displays of the SanityMonitor diagnosis. Both work on the generic structure
produced by diagnose.py ({category: {"checks": [...], "modules": [...]}}),
so any new metric is displayed without touching this file.

verbose:
    0  nothing
    1  only errors
    2  warnings and errors
    3  every check (messages only for warnings / errors)
    4  every check with its message, and the per-module tables
"""

from __future__ import annotations

import math
import shutil

ICON = {"ok": "🟢", "info": "🔵", "warn": "🟡", "error": "🔴"}
TAG = {"ok": "[ OK ]", "info": "[INFO]", "warn": "[WARN]", "error": "[FAIL]"}
COLOR = {"ok": "green", "info": "cyan", "warn": "yellow", "error": "red"}
SHOWN = {1: {"error"}, 2: {"warn", "error"}, 3: {"ok", "info", "warn", "error"},
         4: {"ok", "info", "warn", "error"}}
TITLES = {
    "energy": "Energía", "sampling": "Muestreo", "optimization": "Optimización",
    "gradients": "Derivadas logarítmicas y NTK", "wavefunction": "Función de onda",
    "local_energy": "Energías locales", "magnetization": "Magnetización",
}

MODULE_COLS = [  # (key, header, format)
    ("label", "módulo", "{}"),
    ("size", "nº par.", "{:d}"),
    ("theta", "|θ|", "{:.2e}"),
    ("dp", "|dp|", "{:.2e}"),
    ("rel", "|dp|/|θ|", "{:.2e}"),
    ("share", "% |dp|²", "{:.0%}"),
    ("rel_step", "|Δθ|/|θ|", "{:.2e}"),
    ("o_re", "|O_re|", "{:.2e}"),
    ("o_im", "|O_im|", "{:.2e}"),
    ("dead", "muertos", "{:.0%}"),
]


def _cell(row, key, fmt):
    v = row.get(key)
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def _visible(diagnosis, verbose):
    keep = SHOWN.get(verbose, SHOWN[3])
    out = {}
    for cat, d in diagnosis.items():
        if cat == "overall":
            continue
        checks = [c for c in d["checks"] if c["status"] in keep]
        if checks or (verbose == 4 and d.get("modules")):
            out[cat] = {"checks": checks, "modules": d.get("modules")}
    return out


def _module_columns(rows):
    return [c for c in MODULE_COLS if any(c[0] in r for r in rows)]


# ---------------------------------------------------------------------------
# simple (plain text, safe for log files)
# ---------------------------------------------------------------------------


def display_diagnosis_simple(diagnosis, verbose=3, step=None):
    if verbose <= 0:
        return
    ov = diagnosis["overall"]
    head = f" SANITY MONITOR{'' if step is None else f'  (step {step})'}  {TAG[ov['status']]} {ov['message']} "
    lines = ["", head.center(78, "=")]
    for cat, d in _visible(diagnosis, verbose).items():
        lines.append(f"-- {TITLES.get(cat, cat)}")
        for c in d["checks"]:
            msg = c["message"] if (verbose == 4 or c["status"] in ("warn", "error")) else ""
            lines.append(f"  {TAG[c['status']]} {c['name']:<30s} {c['value']:<28s} {msg}".rstrip())
        if verbose == 4 and d.get("modules"):
            cols = _module_columns(d["modules"])
            widths = [max(len(h), *(len(_cell(r, k, f)) for r in d["modules"])) for k, h, f in cols]
            lines.append("    " + "  ".join(h.ljust(w) for (_, h, _), w in zip(cols, widths)))
            for r in d["modules"]:
                lines.append("    " + "  ".join(_cell(r, k, f).ljust(w) for (k, _, f), w in zip(cols, widths)))
    lines.append("=" * 78)
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# rich (colours, tables)
# ---------------------------------------------------------------------------


def display_diagnosis_sanity_monitor(diagnosis, verbose=3, step=None):
    if verbose <= 0:
        return
    from rich import box
    from rich.console import Console, Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    console = Console(width=min(shutil.get_terminal_size().columns, 140))
    ov = diagnosis["overall"]
    parts = []
    for cat, d in _visible(diagnosis, verbose).items():
        t = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False)
        t.add_column("", width=2)
        t.add_column("métrica", style="bold", ratio=2)
        t.add_column("valor", ratio=2)
        t.add_column("comentario", style="dim", ratio=4)
        for c in d["checks"]:
            msg = c["message"] if (verbose == 4 or c["status"] in ("warn", "error")) else ""
            t.add_row(ICON[c["status"]], escape(c["name"]),
                      f"[{COLOR[c['status']]}]{escape(c['value'])}[/]", escape(msg))
        items = [t]
        if verbose == 4 and d.get("modules"):
            cols = _module_columns(d["modules"])
            mt = Table(box=box.MINIMAL, show_edge=False, header_style="bold cyan")
            for k, h, _ in cols:
                mt.add_column(h, justify="left" if k == "label" else "right")
            for r in d["modules"]:
                mt.add_row(*(escape(_cell(r, k, f)) for k, _, f in cols))
            items.append(mt)
        parts.append(Panel(Group(*items), title=f"[bold]{TITLES.get(cat, cat)}", title_align="left",
                           border_style="steel_blue3"))
    title = f"🔬 SANITY MONITOR{'' if step is None else f'  ·  step {step}'}"
    console.print(Panel(Group(*parts) if parts else "", title=title,
                        subtitle=f"{ICON[ov['status']]} {escape(ov['message'])}",
                        border_style=COLOR[ov["status"]]))
