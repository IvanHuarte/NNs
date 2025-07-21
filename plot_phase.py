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
file_ED = f"Oxalate_" + artifact["model_NN"]["name"] + f"_SSF_strength_{strength:2f}_theta_{theta:2f}_phi_{phi:2f}_ED.txt"
files = [file]
