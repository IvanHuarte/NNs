"""
Folder layout and file labels of a simulation.

    <write_folder_sim>/<CM>/<NN>_<template>/Size_<Lx>x<Ly>/<schedule>/UUID_<uuid>/
        <CM>_<NN>_simulation_<Lx>x<Ly><cparams>_date_<date>_UUID_<uuid>...   (sim)
        <CM>_<NN>_xED_<Lx>x<Ly><cparams>_date_<date>_UUID_<uuid>...          (ED)
        <CM>_<NN>_results_<Lx>x<Ly><cparams>_date_<date>_UUID_<uuid>.json    (json)

One UUID folder per size: it holds every point of a config1 '*_list' sweep,
told apart by <cparams>. Analysis scripts glob '*_results_*.json' inside
'UUID_*' folders and parse <cparams> (e.g. '_J1J2_<J1>_<J2>_'), so these
formats must stay stable.

Labels are built ONCE per simulation point (make_labels) and stored in the
results JSON under "labels"; analysis scripts read them back with
labels_from_artifact instead of rebuilding them (the date would differ).
"""

import re
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime

from NN_module.schedule.utils import get_schedule_label

DATE_FORMAT = "%Y%m%dT%H%M%S"


# ============================================================
# Coupling-model parameters -> (filename fragment, plot title fragment)
# ============================================================


def _fields_couplings(field_ops, fields, coupling_ops, couplings):
    """'_X_Z_0.5_0.1__ZZ_1.0_' style fragment (IsingSquare)."""
    flat_fields = "".join(f"{op}_" for op in field_ops)
    field_values = "".join(f"{v}_" for v in fields)
    flat_couplings = "".join(f"{op}_" for op in coupling_ops)
    coupling_values = "".join(f"{v}_" for v in couplings)
    title = "".join(
        f"{op}:{v}  "
        for op, v in zip(
            list(field_ops) + list(coupling_ops), list(fields) + list(couplings)
        )
    )
    return f"_{flat_fields}{field_values}_{flat_couplings}{coupling_values}", title


def _oxalate(p):
    if "couplings" in p:
        strength, theta, phi = p["couplings"]
    else:  # before ModelFactory.preprocess_params
        strength, theta, phi = p["strength"], p["theta"], p["phi"]
    return (
        f"_strength_{strength}_theta_{theta}_phi_{phi}",
        r"$a = %.1f$  $\theta = %.1f$  $\phi = %.1f$" % (strength, theta, phi),
    )


def _ising_square(p):
    # Just 1 coupling for Ising
    return _fields_couplings(p["ops"][0], p["fields"], p["ops"][1][:1], [p["J"]])


def _long_range(p):
    J, alpha, f = p["J"], p["alpha"], p["fields"]
    return (
        f"_J_{J}_alpha_{alpha}_XZ_{f[0]}_{f[1]}",
        r"$J=%.2f$  $\alpha=%.1f$  $XZ=(%.1f,%.1f)$" % (J, alpha, f[0], f[1]),
    )


def _j1j2_square(p):
    J1, J2, f = p["J1"], p["J2"], p["fields"]
    return (
        f"_J1J2_{J1}_{J2}_XYZ_{f[0]}_{f[1]}_{f[2]}",
        r"$J1=%.2f$  $J2=%.2f$  $XYZ=(%.1f,%.1f,%.1f)$" % (J1, J2, f[0], f[1], f[2]),
    )


def _ssm_square(p):
    J, Jp, f = p["J"], p["Jp"], p["fields"]
    return (
        f"_JJp_{J}_{Jp}_XYZ_{f[0]}_{f[1]}_{f[2]}",
        r"$J=%.2f$  $J_p=%.2f$  $XYZ=(%.1f,%.1f,%.1f)$" % (J, Jp, f[0], f[1], f[2]),
    )


def _generic(p):
    """Fallback for models without a registered labeler: every parameter."""

    def fmt(v):
        if isinstance(v, (list, tuple)):
            return "_".join(fmt(x) for x in v)
        return str(v)

    items = [(k, v) for k, v in p.items() if not k.endswith("_list")]
    return (
        "".join(f"_{k}_{fmt(v)}" for k, v in items),
        "  ".join(f"{k}={fmt(v)}" for k, v in items),
    )


# Register here the label of new coupling models (otherwise _generic is used)
CM_LABELS = {
    "Oxalate": _oxalate,
    "IsingSquare": _ising_square,
    "LRChain": _long_range,
    "LRSquare": _long_range,
    "J1J2Square": _j1j2_square,
    "SSMSquare": _ssm_square,
}


def cm_label(cm_setup):
    """(filename fragment, title fragment) of the coupling-model parameters."""
    labeler = CM_LABELS.get(cm_setup["name"], _generic)
    return labeler(cm_setup["params"])


# ============================================================
# Labels of one simulation point
# ============================================================


@dataclass(frozen=True)
class SimLabels:
    sim: str  # parameters, sampler state, callback plot/txt
    ED: str  # exact-diagonalization vectors
    json: str  # main results artifact
    title: str  # plot title ("Callback  <model>  <params>  (<size>)")

    def as_dict(self):
        """Keys stored in the results JSON under 'labels'."""
        return {
            "sim_label": self.sim,
            "ED_label": self.ED,
            "json_label": self.json,
            "title_label_callback": self.title,
        }

    @property
    def plot_title(self):
        """Title without the 'Callback' prefix, LaTeX spaced (modphase/SSF plots)."""
        return self.title.replace("Callback", "").lstrip().replace(" ", "\\quad")


def make_labels(cm_setup, nn_name, sim_uuid, date=None):
    """Build the labels of a simulation point. Call it once per point."""
    date = date or datetime.now().strftime(DATE_FORMAT)
    model_label = cm_setup["name"] + "_" + nn_name
    size = f"{cm_setup['size'][0]}x{cm_setup['size'][1]}"
    cparams, title_params = cm_label(cm_setup)
    tail = f"{cparams}_date_{date}_UUID_{sim_uuid}"

    return SimLabels(
        sim=f"{model_label}_simulation_{size}{tail}",
        ED=f"{model_label}_xED_{size}{tail}",
        json=f"{model_label}_results_{size}{tail}",
        title=f"Callback  {model_label}  {title_params}  ({size})",
    )


def make_title(cm_setup, nn_name):
    """Plot title only (no date/uuid needed)."""
    return make_labels(cm_setup, nn_name, sim_uuid="").title


_DATE_RE = re.compile(r"_date_(\d{8}T\d{6})_UUID_")


def labels_from_artifact(artifact, artifact_path=None):
    """
    Labels of a saved simulation. Uses the stored 'labels' block; for old
    artifacts without it they are rebuilt, taking the date from the
    artifact file name when given (otherwise the date will not match).
    """
    if "labels" in artifact:
        lab = artifact["labels"]
        return SimLabels(
            sim=lab["sim_label"],
            ED=lab["ED_label"],
            json=lab["json_label"],
            title=lab["title_label_callback"],
        )

    date = None
    if artifact_path is not None:
        match = _DATE_RE.search(str(artifact_path))
        date = match.group(1) if match else None
    if date is None:
        print("Warning: artifact without 'labels' nor date; labels use today's date")

    metadata = artifact.get("metadata", {})
    return make_labels(
        artifact["CM"],
        artifact["NN"]["name"],
        metadata.get("uuid"),
        date=date,
    )


# ============================================================
# Folders
# ============================================================


def get_string_from_nnsetup(tree, level=0):
    if not isinstance(tree, dict) or not tree:
        return ""

    names = []

    for module, submodules in tree.items():
        children = get_string_from_nnsetup(submodules, level + 1)
        separator = "_" * (level + 1)

        if children:
            names.append(f"{module}{separator}{children}")
        else:
            names.append(module)

    return "_".join(names)


def model_folder(write_folder_sim, cm_name, nn_name, nn_evol_setup):
    """<write_folder_sim>/<CM>/<NN>_<template of the last stage>/"""
    last_stage = next(reversed(nn_evol_setup))
    nn_label = nn_name + "_" + get_string_from_nnsetup(
        nn_evol_setup[last_stage]["template"]
    )
    return write_folder_sim + cm_name + "/" + nn_label + "/"


@dataclass(frozen=True)
class RunPaths:
    uuid: str
    write_folder: str  # .../Size_LxL/<schedule>/UUID_<uuid>/

    @property
    def parent(self):
        """Folder that holds the UUID folders (schedule level)."""
        return self.write_folder.rstrip("/").rsplit("/", 1)[0] + "/"


def make_run_paths(config, cm_name, nn_name, nn_evol_setup, size, sim_uuid=None):
    """Folder of the simulations of one size (shared by a parameter sweep)."""
    sim_uuid = sim_uuid or str(_uuid.uuid4())[:8]
    folder = (
        model_folder(config["write_folder_sim"], cm_name, nn_name, nn_evol_setup)
        + f"Size_{size[0]}x{size[1]}/"
        + get_schedule_label(config["schedule"]["learning_rate"])
        + f"/UUID_{sim_uuid}/"
    )
    return RunPaths(uuid=sim_uuid, write_folder=folder)

