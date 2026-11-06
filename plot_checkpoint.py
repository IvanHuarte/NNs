#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python
import numpy as np
import jax

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)

import os
import argparse
import json
import sys
import pathlib
import orbax.checkpoint as ocp

from NN_module.observables import modphase
from NN_module.dynamic_plots import plot_modphase
from NN_module.callback.utils import checkpoint_callback
from NN_module.label_utils import labels_from_artifact
from NN_module.saveNload import load_vstate


parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--artifact_path",
    type=str,
    required=True,
    help="Path al artefacto principal que recoge los resultados de la simulacion",
)
parser.add_argument(
    "-m",
    "--modphase",
    type=str,
    required=False,
    default=None,
    help="Path al artefacto principal que recoge los resultados de la simulacion",
)

args = parser.parse_args()

path_artifact = args.artifact_path
do_modphase = args.modphase if args.modphase is not None else False
write_folder = os.path.dirname(path_artifact) + "/"

no_artifact = True if path_artifact.endswith(".orbax") else False


x_ED = None
modphase_ED = None
artifact = None
# Load artifact and simulation params if provided
if not no_artifact:
    with open(path_artifact, "r") as f:
        artifact = json.load(f)

    size = artifact["CM"]["size"]
    N = int(np.array(size).prod())
    cm_model_name = artifact["CM"]["name"]
    nn_model_name = artifact["NN"]["name"]
    cm_model_setup = artifact["CM"]
    nn_model_setup = artifact["NN"]
    sim_uuid = artifact["metadata"]

    sim_label = labels_from_artifact(artifact, path_artifact).sim
        
    # Reference state: irrep ground state if available, else the global one
    # ('x_ED' in older artifacts)
    arts = artifact["_artifacts"]
    x_ED_path = arts.get("x_ED_irrep") or arts.get("x_ED_global") or arts.get("x_ED")
    if x_ED_path is not None:
        x_ED = np.loadtxt(x_ED_path, dtype=complex)
        modphase_ED = modphase(x_ED)

else:
    sim_label = "Broken simulation"

checkpoint_path = path_artifact if no_artifact else artifact["_artifacts"]["checkpoint"]

checkpointer = ocp.CheckpointManager(
    checkpoint_path,
    {
        "metrics": ocp.PyTreeCheckpointer()
    },
)

write_folder = (
    str(pathlib.Path(path_artifact).parent) + "/" + "Checkpoint_figures" + "/"
)
os.makedirs(write_folder, exist_ok=True)


steps = os.listdir(checkpoint_path)
checkPath = [str(folder) for folder in pathlib.Path(checkpoint_path).iterdir()]

# Sort steps and corresponding paths
steps = [int(step) for step in steps]
sort_idx = np.argsort(steps)
steps = [steps[idx] for idx in sort_idx]
checkPath = [checkPath[idx] for idx in sort_idx]

energy = []
vscore = []

for step, path in zip(steps, checkPath):

    metrics = checkpointer.restore(
        step,
        items={"metrics": None},
    )["metrics"]

    print(metrics)
    energy.append(metrics["energy"])
    vscore.append(metrics["vscore"])

    if do_modphase:

        params_path = path + "/" + "parameters"
        sampler_path = path + "/" + "sampler_state"
        reload_vstate = load_vstate(
            artifact, params_path=params_path, sampler_path=sampler_path
        )
        modphase_vs = modphase(reload_vstate)

        fig, ax = plot_modphase(
            artifact, modphase_vs=modphase_vs, modphase_ED=modphase_ED, step=step
        )
        fig.savefig(
            write_folder + f"ModPhase_checkpoint_{sim_label}_step_{step}.jpeg",
            dpi=600,
            bbox_inches="tight",
        )
        fig.clear()


fig2, ax2 = checkpoint_callback(artifact, steps, energy, vscore, path_artifact)
file_path = write_folder + f"Callback_checkpoint_" + sim_label
figure_path = file_path + ".jpeg"

fig2.tight_layout()
fig2.savefig(figure_path, dpi=600, bbox_inches="tight")

if not no_artifact:
    artifact["_artifacts"]["checkpoint_figures"] = write_folder
    with open(path_artifact, "w") as f:
        json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)
