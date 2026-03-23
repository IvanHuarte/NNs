#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import os
import argparse
import json
import sys
from pathlib import Path

from NN_module.saveNload import load_vstate
from NN_module.correlations import correlations_ED, correlations_vstate
from NN_module.label_utils import get_filenames_from_settings


def get_nk_graph(cm_name, size, pbc=True):

    if cm_name == "Oxalate":
        return nk.graph.Triangular(size, pbc=pbc)
    elif cm_name == "J1J2Square":
        return nk.graph.cm_name(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            size,
            site_offsets=[[0, 0]],
            pbc=pbc,
        )
    elif cm_name == "Chain":
        return nk.graph.Chain(size[0], pbc=pbc)


# Añadir el directorio chebyoxa al path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA"))
import chebyoxa.utils as utils
from chebyoxa_functions import *

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
write = os.path.dirname(path_artifact) + "/"

with open(path_artifact, "r") as f:
    artifact = json.load(f)

size = artifact["CM"]["size"]
N = int(np.array(size).prod())

cm_name = artifact["CM"]["name"]
nn_name = artifact["NN"]["name"]
model_label = cm_name + " " + nn_name

kwargs = {"size": size, **artifact["CM"], **artifact["NN"]}

uuid = artifact["metadata"]["uuid"]

sim_label, ED_label, _, _ = get_filenames_from_settings(
    artifact["CM"], {"name": nn_name, "setup": {}}, sim_uuid=uuid
)


file = f"{sim_label}_SSF_vstate.txt"
file_ED = f"{ED_label}_SSF_ED.txt"
files = [file]

# Check exact diagonalization mode
SSF_label = [artifact["NN"]["name"].split("_")[0]]
if artifact["results"]["E_ED"] is not None:
    exact_diag = True
    SSF_label.append("E_ED")
    files.append(file_ED)
else:
    exact_diag = False

if os.path.isfile(write + file_ED):
    exact_diag = False

# Some Sebas's stuff
N_Q_A = 48 * 3
N_Q_B = 48 * 3
graph = get_nk_graph(cm_name, size, pbc=True)
n_spins = size[0] * size[1]
qs_mapping, direct_qs = utils.get_q_mesh(N_Q_A, N_Q_B, graph)
all_sites = list(zip(*np.triu_indices(n_spins)))

# Load vstate   and check it exists
if not os.path.isfile(artifact["_artifacts"]["parameters"]) and not os.path.isdir(
    artifact["_artifacts"]["parameters"]
):
    print(
        f"vstate parameters file: {artifact['_artifacts']['parameters']} not available!"
    )
    sys.exit(1)

vstate = load_vstate(artifact)

if os.path.isfile(write + file_ED):
    print(f"SSF ED already done!")
    exact_diag = False

# Calculate correlations and structure factor
correlations_list = []

corr_NN = correlations_vstate(vstate)
correlations_list.append(corr_NN)
print(f"vstate correlations DONE\n\n")


if exact_diag:
    x_ED_path = artifact["_artifacts"]["x_ED"]
    x_ED = np.loadtxt(x_ED_path, dtype="complex")
    corr_ED = correlations_ED(size, x_ED)
    correlations_list.append(corr_ED)
    print(f"ED correlations DONE\n\n")

art_label = ["OPT", "ED"]
artifact["_artifacts"]["SSF"] = {}

for i, corr in enumerate(correlations_list):

    print(SSF_label[i])
    SSF = calculate_structure_factor(corr, graph, direct_qs, qs_mapping, N_Q_A, N_Q_B)
    np.savetxt(write + files[i], SSF)
    print("Done\n\n")

for i, f in enumerate(files):
    artifact["_artifacts"]["SSF"][art_label[i]] = write + f

with open(path_artifact, "w") as f:
    json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)
