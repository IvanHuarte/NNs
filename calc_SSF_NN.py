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

from NN_module.NN_utils import load_vstate
from NN_module.correlations import correlations_ED, correlations_vstate

# Añadir el directorio chebyoxa al path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA"))
import chebyoxa.utils as utils
from chebyoxa_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--artifact_path', type=str, required=True, help='Path al artefacto principal que recoge los resultados de la simulacion')
parser.add_argument('--write_folder', type=str, required=True, help='Path donde escribir los SSF')
parser.add_argument('--explore_mode', type=str, help='Modo para discriminar simulaciones con los mismos parametros')
args=parser.parse_args()

path_artifact= args.artifact_path
write_folder_ssf = args.write_folder
explore_mode = args.explore_mode

with open(path_artifact,'r') as f:
    artifact = json.load(f)

strength = artifact['coupling_model']['strength']
size = artifact['lattice']['size']
theta = artifact['coupling_model']['theta']
phi = artifact['coupling_model']['phi']

write_folder_SSF = write_folder_ssf #+ f"Oxalate_size_{size[0]}x{size[1]}/"

file = f"Oxalate_" + artifact["model_NN"]["name"] + f"_SSF_strength_{strength:2f}_theta_{theta:2f}_phi_{phi:2f}"
file_ED = f"Oxalate_" + artifact["model_NN"]["name"] + f"_SSF_strength_{strength:2f}_theta_{theta:2f}_phi_{phi:2f}_ED"
files = [file]

# Check exact diagonalization mode
SSF_label=[artifact["model_NN"]["name"]]
if artifact["results"]["E_ED"] is not None:
    exact_diag = True
    SSF_label.append('E_ED')
    files.append(file_ED)
else:
    exact_diag = False

if os.path.isfile(write_folder_SSF + file_ED):
    exact_diag = False

#Some Sebas's stuff
A = 0.2
N_Q_A = 48 * 3
N_Q_B = 48 * 3
graph = nk.graph.Triangular(size, pbc=True)
n_spins = size[0] * size[1]
qs_mapping, direct_qs = utils.get_q_mesh(N_Q_A, N_Q_B, graph)
all_sites = list(zip(*np.triu_indices(n_spins)))

# Load vstate
if not os.path.isfile(artifact["results"]["vstate"]):
    print(f"vstate parameters from size: {size} theta:{theta} phi:{phi} not available!")
    exit()

# artifact["model_NN"]["activation"] = ast.literal_eval(artifact["model_NN"]["activation"])
# artifact["model_NN"]["dense_dim"] = ast.literal_eval(artifact["model_NN"]["dense_dim"])

vstate=load_vstate(artifact)

# Verify there is no previous simulations, to earn time
print(f"\n************size: {size} theta: {theta:1f}  phi: {phi:1f} ************\n\n")
if explore_mode:                    # If True checks sucessive files and assign a new one
    
    i=1
    file_temp = file + f"_{i}.txt"
    while(os.path.isfile(write_folder_SSF + file_temp)):
        file_temp = file + f"_{i}.txt"
        i+=1
        print(f"_{i}.txt")
    
    files[0] += f"_{i}.txt"

if os.path.isfile(write_folder_SSF + file_ED): 
    print(f"SSF ED size: {size} theta:{theta} phi:{phi} already done!")
    exact_diag =False

# Calculate correlations and structure factor
correlations_list = []

corr_NN = correlations_vstate(vstate)
correlations_list.append(corr_NN)
print(f"vstate correlations DONE\n\n")

x_ED_path = artifact['_artifacts']["x_ED"]
if exact_diag:
    x_ED = np.loadtxt(x_ED_path, dtype = np.complex64)
    corr_ED = correlations_ED(size, x_ED)
    correlations_list.append(corr_ED)
    print(f"ED correlations DONE\n\n")

art_label=['OPT','ED']
artifact['_artifacts']['SSF'] = {}
for i, corr in enumerate(correlations_list):

    print(SSF_label[i])
    SSF = calculate_structure_factor(corr, graph, direct_qs, qs_mapping, N_Q_A, N_Q_B)
    np.savetxt(write_folder_SSF + files[i], SSF)
    print("Done\n\n")

for i, f in enumerate(files):
    artifact['_artifacts']['SSF'][art_label[i]] = write_folder_SSF + f

with open(path_artifact,"w") as f:
    json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)

