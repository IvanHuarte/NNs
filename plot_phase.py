#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import os
import argparse
import json
import ast
import sys
from pathlib import Path

from NN_module.saveNload import load_vstate
from NN_module.label_utils import get_filenames_from_settings
from NN_module.observables import modphase_extended
from NN_module.correlations import correlations_ED, correlations_vstate

# Añadir el directorio chebyoxa al path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA"))

#### MAIN ####

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--artifact_path",
    type=str,
    required=True,
    help="Path al artefacto principal que recoge los resultados de la simulacion",
)
parser.add_argument(
    "-w", "--write_folder", type=str, help="Path donde escribir los SSF"
)
parser.add_argument(
    "-exp",
    "--explore_mode",
    type=str,
    help="Modo para discriminar simulaciones con los mismos parametros",
)
args = parser.parse_args()

path_artifact = args.artifact_path
write_folder = (
    args.write_folder
    if args.write_folder is not None
    else os.path.dirname(path_artifact) + "/"
)
explore_mode = args.explore_mode

# Load artifact and simulation params
with open(path_artifact, "r") as f:
    artifact = json.load(f)

# Get main result's modulus and phase 
vs_file = artifact["_artifacts"]["modphase"]["vstate"]
mod_vs, phase_vs = np.loadtxt(vs_file)
stats_vs = artifact["results"]["modphase"]["vstate"]

ED_file = None
if "xED" in artifact["_artifacts"]["modphase"]:
    ED_file = artifact["_artifacts"]["modphase"]["xED"]
    mod_ED, phase_ED = np.loadtxt(ED_file)
    stats_ED = artifact["results"]["modphase"]["xED"]


if not "modphase" in artifact["_artifacts"]:
    print("ERROR: Artifact has no modulus and phase files")
    sys.exit(1, f"Exiting...")

size = artifact["CM"]["size"]
N = int(np.array(size).prod())


cm_model_name = artifact["CM"]["name"]
nn_model_name = artifact["NN"]["name"]
cm_model_setup = artifact["CM"]
nn_model_setup = artifact["NN"]
sim_uuid = artifact["metadata"]


kwargs = {"size": size, **artifact["CM"], **artifact["NN"]["setup"]}

sim_label, _, _, callback = get_filenames_from_settings(
    cm_model_setup, {"name": nn_model_name, "setup": {}}
)
title = callback.replace("Callback", "").lstrip().replace(" ", "\\quad")

filename = f"Modphase_plot_{sim_label}"

if mod_vs.shape[-1] != 2**N:
    vstate_label = f"Approximated ({artifact['SIM']['sampler']['nsamples']} samples)"
else:
    vstate_label = "Exact"

# Plot
# If ED exists
if ED_file is not None:

    markersize = {"[]"}

    _, ax = plt.subplots(4, 1, figsize=[15, 10])

    ax[0].set_title(r"$Modulus\;and\;Phase\qquad %s$" % (title), fontsize=10)
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"$Modulus$")
    ax[0].set_ylim(-0.00001, max(max(mod_ED), max(mod_vs)) * 9 / 8)
    ax[0].plot(mod_vs, alpha=0.6, color="r", label=f"vstate")
    ax[0].plot(mod_ED, alpha=0.6, label=f"ED")
    ax[0].legend()

    ax[1].set_xticks([])
    ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax[1].set_ylabel(r"$Phase \;vstate$")
    ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax[1].plot(
        phase_vs, alpha=0.15, ls="", marker="o", ms=0.1, color="r", label="vstate"
    )
    ax[1].legend(loc="upper right")

    ax[2].set_xlabel(r"$C_i$")
    ax[2].set_ylabel(r"$Phase \;ED$")
    ax[2].set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax[2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[2].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax[2].plot(phase_ED, alpha=0.15, ls="", marker="o", ms=0.1, label="ED")
    ax[2].legend(loc="upper right")

    ax[3].set_xlabel(r"$Phase\;(radians)$")
    ax[3].set_ylabel(r"$Phase \;histogram$")
    ax[3].hist(
        phase_ED,
        bins=1000,
        range=(-np.pi, np.pi),
        density=True,
        alpha=0.7,
        label=f"ED  ({stats_ED['type']})",
    )
    ax[3].hist(
        phase_vs,
        bins=1000,
        range=(-np.pi, np.pi),
        color="r",
        density=True,
        alpha=0.7,
        label=f"vstate ({stats_vs['type']})",
    )
    ax[3].text(
        0.8,
        0.7,
        r"$\varphi_{ED}=%.2f \pm %.2f$"
        % (stats_ED["phase"]["mean"], stats_ED["phase"]["std"])
        + "\n"
        + r"$\varphi_{vs}=%.2f \pm %.2f$"
        % (stats_vs["phase"]["mean"], stats_vs["phase"]["std"]),
        transform=ax[3].transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.4),
    )

    transform = mtransforms.blended_transform_factory(ax[3].transData, ax[3].transAxes)
    if stats_ED["peaks"] is not None:
        for peak in stats_ED["peaks"]["values"]:
            ax[3].text(
                peak - 0.1,
                0.9,
                r"%.2f" % peak,
                color="b",
                transform=transform,
                fontsize=8,
                alpha=0.7,
            )

    if stats_vs["peaks"] is not None:
        for peak in stats_vs["peaks"]["values"]:
            ax[3].text(
                peak - 0.1,
                0.9,
                r"%.2f" % peak,
                color="b",
                transform=transform,
                fontsize=8,
                alpha=0.7,
            )
    ax[3].legend()
    plt.tight_layout()
    plt.savefig(write_folder + filename + ".png", dpi=600, bbox_inches="tight")

else:

    _, ax = plt.subplots(3, 1, figsize=[13, 9])

    ax[0].set_title(r"$Modulus\;and\;Phase\qquad %s$" % (title), fontsize=10)
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"$Modulus$")
    ax[0].set_ylim(-0.00001, max(mod_vs) * 9 / 8)
    ax[0].plot(mod_vs, alpha=0.6, color="r", label="vstate")
    ax[0].legend()

    ax[1].set_xlabel(r"$C_i$")
    ax[1].set_xticks([])
    ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax[1].set_ylabel(r"$Phase \;vstate$")
    ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax[1].plot(
        phase_vs, alpha=0.25, ls="", marker="o", ms=0.9, color="r", label="vstate"
    )
    ax[1].legend(loc="upper right")

    ax[2].set_xlabel(r"$Phase\;(radians)$")
    ax[2].set_ylabel(r"$Phase \;histogram$")
    ax[2].hist(
        phase_vs,
        bins=1000,
        range=(-np.pi, np.pi),
        color="r",
        density=True,
        alpha=0.7,
        label="vstate",
    )
    ax[2].text(
        0.8,
        0.7,
        r"$\varphi_{vs}=%.2f \pm %.2f$"
        % (stats_vs["phase"]["mean"], stats_vs["phase"]["std"]),
        fontsize=10,
        transform=ax[2].transAxes,
        bbox=dict(facecolor="white", alpha=0.4),
    )
    transform = mtransforms.blended_transform_factory(ax[2].transData, ax[2].transAxes)

    if stats_vs["peaks"] is not None:
        for peak in stats_vs["peaks"]["values"]:
            ax[2].text(
                peak - 0.1,
                0.9,
                r"%.2f" % peak,
                color="b",
                transform=transform,
                fontsize=8,
                alpha=0.7,
            )
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(write_folder + filename + ".png", dpi=600, bbox_inches="tight")


artifact["_artifacts"]["modphase_plot"] = write_folder + filename + ".png"
with open(path_artifact, "w") as f:
    json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)
