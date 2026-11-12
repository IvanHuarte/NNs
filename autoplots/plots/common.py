"""Labels and saving shared by the aplot figures."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # NNs

LABELS = {
    "E": r"$E$",
    "E_spin": r"$E$ ($S=\sigma/2$)",
    "E_site": r"$E/N$",
    "E_site_spin": r"$E/N$ ($S=\sigma/2$)",
    "E50": r"$E_{50}$",
    "E_re": r"$E_{\rm re}$",
    "E_ref": r"$E_{\rm ref}$",
    "err_rel": r"$(E-E_{\rm ref})/|E_{\rm ref}|$",
    "gap": r"$\Delta$ ($S=\sigma/2$)",
    "vscore": "V-score",
    "vscore50": r"V-score$_{50}$",
    "var": r"Var$(E)/N$",
    "J1": r"$J_1$",
    "J2": r"$J_2$",
    "Jp": r"$J_p$",
    "L": r"$L$",
    "N": r"$N$",
    "s_it": "s / epoch",
    "time_h": "time (h)",
    "energy": r"$E$",
    "variance": r"Var$(E)$",
    "step": "epoch",
}


def label(name):
    return LABELS.get(name, name)


def git_version():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def save(fig, data, base, fmt, cmd):
    """
    base.<fmt> with the command and git version in its metadata, and base.csv
    with the plotted values (first line: the same provenance).
    """
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"aplot {cmd} | git {git_version()}"
    key = {"png": "Description", "pdf": "Subject", "svg": "Description"}.get(fmt)
    fig.savefig(f"{base}.{fmt}", bbox_inches="tight", metadata={key: stamp} if key else None)
    if data is not None:
        with open(f"{base}.csv", "w") as f:
            f.write(f"# {stamp}\n")
            data.to_csv(f, index=False)
    print(f"[aplot] {base}.{fmt}")
