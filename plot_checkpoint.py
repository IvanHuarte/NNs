#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

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
from .plot_phase import plot_modphase

# Añadir el directorio chebyoxa al path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--artifact_path",
    type=str,
    required=True,
    help="Path al artefacto principal que recoge los resultados de la simulacion",
)

args = parser.parse_args()

path_artifact = args.artifact_path
write_folder = os.path.dirname(path_artifact) + "/"


# Load artifact and simulation params
with open(path_artifact, "r") as f:
    artifact = json.load(f)


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

filename = f"Checkpoint_callback_{sim_label}"



artifact["_artifacts"]["checkpoint_callback"] = write_folder + filename + ".jpeg"
with open(path_artifact, "w") as f:
    json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)
