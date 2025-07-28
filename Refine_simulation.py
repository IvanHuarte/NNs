#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from netket.operator.spin import sigmaz
import optax
import json
import argparse
import time
import ast
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()

# Añadir los directorios necesarios
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA/VA_project/src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "Transformers/transformer_LR_WF_public"))

# Importar módulos necesarios
from VA_project.model.model import OxalateJKGamma
from VA_project.engine.runners import Runner
from NN_module.sim_utils import (
    save_results, dump_callback, init_model, architecture_label,
    get_filenames_from_settings, load_vstate
)
from NN_module.NN_utils import (
    activation_dict, scheduler_initializer, 
    phase_stats_ED, phase_stats_vstate,
    modphase
)
from NN_module.ST_utils import compare_params, masked_optimizer
from transformer_LR_WF.utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-a",'--artifact_path', type=str, required=True, help='Path al artefacto principal que recoge los resultados de la simulacion')
args=parser.parse_args()
path_artifact= args.artifact_path

with open(path_artifact,'r') as f:
    artifact = json.load(f)

# Get filenames in base
kwargs={
    'size':artifact['lattice']['size']
    **artifact['coupling_model'],
    **artifact['model_NN']['setup']
} 
sim_label, _, json_label, title_label_callback = get_filenames_from_settings('Oxalate', artifact["model_NN"]["name"], **kwargs)

# Refinement filenames

write_folder = os.path.dirname(path_artifact) + "/Refinements/"
if not os.path.exists(write_folder):
    os.makedirs(write_folder)

if os.path.isfile(write_folder + json_label + ".json"):
    i=1
    file_temp = json_label + f"_{i}.json"
    while(os.path.isfile(write_folder + file_temp)):
        i+=1
        file_temp = json_label + f"_{i}.json"

    sim_label+=f"_{i}"
    json_label+=f"_{i}"


# Load vstate....
vstate=load_vstate(artifact)
