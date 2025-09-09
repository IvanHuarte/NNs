#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import os
import argparse
import json
import ast
import sys
from pathlib import Path

from NN_module.sim_utils import load_vstate
from NN_module.correlations import correlations_ED, correlations_vstate
from NN_module.label_utils import get_filenames_from_settings

def get_nk_graph(lattice, size, pbc=True):

    if lattice == 'Triangular':
        return nk.graph.Triangular(size, pbc=pbc)
    elif lattice == 'Chain':
        return nk.graph.Chain(size[0], pbc=pbc)
    elif lattice == 'Square':
        return nk.graph.Lattice(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            size,
            site_offsets=[[0, 0]],
            pbc=pbc
        )

# Añadir el directorio chebyoxa al path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA"))
import chebyoxa.utils as utils
from chebyoxa_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("-a", '--artifact_path', type=str, required=True, help='Path al artefacto principal que recoge los resultados de la simulacion')
parser.add_argument("-exp", '--explore_mode', type=str, help='Modo para discriminar simulaciones con los mismos parametros')
args=parser.parse_args()

path_artifact= args.artifact_path
explore_mode = args.explore_mode
write = os.path.dirname(path_artifact) + "/"

with open(path_artifact,'r') as f:
    artifact = json.load(f)

size = artifact['lattice']['size']
N = int(np.array(size).prod())

model_label = artifact['model_label']
cm_name = model_label.split("_",1)[0]
nn_name = model_label.split("_",1)[1]

kwargs ={
    'size':size,
    **artifact['coupling_model'],
    **artifact['model_NN']['setup']
}

sim_label, ED_label, _, _ = get_filenames_from_settings(cm_name, nn_name, **kwargs )


file = f"SSF_vstate_{sim_label}"
file_ED = f"SSF_ED_{ED_label}.txt"
files = [file]

# Check exact diagonalization mode
SSF_label=[artifact["model_NN"]["name"]]
if artifact["results"]["E_ED"] is not None:
    exact_diag = True
    SSF_label.append('E_ED')
    files.append(file_ED)
else:
    exact_diag = False

if os.path.isfile(write + file_ED):
    exact_diag = False

#Some Sebas's stuff
A = 0.2
N_Q_A = 48 * 3
N_Q_B = 48 * 3
lattice = artifact['lattice']['name']
graph = get_nk_graph(lattice, size, pbc=True)
n_spins = size[0] * size[1]
qs_mapping, direct_qs = utils.get_q_mesh(N_Q_A, N_Q_B, graph)
all_sites = list(zip(*np.triu_indices(n_spins)))

# Load vstate   and check it exists
if not os.path.isfile(artifact["results"]["vstate"]):
    print(f"vstate parameters file: {artifact['results']['vstate']} not available!")
    sys.exit(1)

# artifact["model_NN"]["activation"] = ast.literal_eval(artifact["model_NN"]["activation"])
# artifact["model_NN"]["dense_dim"] = ast.literal_eval(artifact["model_NN"]["dense_dim"])

vstate=load_vstate(artifact)

# Verify there is no previous simulations, to earn time
if explore_mode:                    # If True checks sucessive files and assign a new one
    
    i=1
    file_temp = file + f"_{i}.txt"
    while(os.path.isfile(write + file_temp)):
        file_temp = file + f"_{i}.txt"
        i+=1
        print(f"_{i}.txt")
    
    files[0] += f"_{i}.txt"


if os.path.isfile(write + file_ED): 
    print(f"SSF ED already done!")
    exact_diag =False

# Calculate correlations and structure factor
correlations_list = []

corr_NN = correlations_vstate(vstate)
correlations_list.append(corr_NN)
print(f"vstate correlations DONE\n\n")


if exact_diag:
    x_ED_path = artifact['_artifacts']["x_ED"]
    x_ED = np.loadtxt(x_ED_path, dtype = np.complex64)
    corr_ED = correlations_ED(size, x_ED)
    correlations_list.append(corr_ED)
    print(f"ED correlations DONE\n\n")

art_label=['OPT','ED']
artifact['_artifacts']['SSF'] = {}

for i, corr in enumerate(correlations_list):

    print(SSF_label[i])
    SSF = calculate_structure_factor(corr, graph, direct_qs, qs_mapping, N_Q_A, N_Q_B)
    np.savetxt(write + files[i], SSF)
    print("Done\n\n")

for i, f in enumerate(files):
    artifact['_artifacts']['SSF'][art_label[i]] = write + f

with open(path_artifact,"w") as f:
    json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)
