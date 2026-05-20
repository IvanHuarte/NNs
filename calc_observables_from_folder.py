#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import os
import argparse
import json
import netket as nk
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
from NN_module.saveNload import load_vstate
from NN_module.observables import calc_all_observables_vs, calc_all_observables_ED


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--directory", type=str, required=True, help="Directorio padre"
)
args = parser.parse_args()
parent = args.directory


parent_folder = parent if parent.endswith("/") else parent + "/"

files_list = []
for root, dirs, files in os.walk(parent_folder):

    for file in files:
        files_list.append(os.path.join(root, file))

files_list = [file for file in files_list if ".json" in file]

for file in files_list:

    print(f"file --> {file}")
    # Load artifacts and simulation params
    with open(file, "r") as f:
        art = json.load(f)

    vstate = load_vstate(art)
    S_renyi, m, ms, m2, ms2 = calc_all_observables_vs(vstate)

    art["results"]["S_renyi"] = float(S_renyi)
    art["results"]["m"] = float(m)
    art["results"]["ms"] = float(ms)
    art["results"]["m2"] = float(m2)
    art["results"]["ms2"] = float(ms2)

    if art["results"]["E_ED"] is not None:
        file_ED = art["_artifacts"]["x_ED"]
        x_ED = np.loadtxt(file_ED, dtype=complex)
        m_ED, ms_ED, m2_ED, ms2_ED = calc_all_observables_ED(x_ED)

        art["results"]["m_ED"] = float(m_ED)
        art["results"]["ms_ED"] = float(ms_ED)
        art["results"]["m2_ED"] = float(m2_ED)
        art["results"]["ms2_ED"] = float(ms2_ED)

    with open(file, "w") as f:
        json.dump(art, f, separators=(",", ":"), sort_keys=True, indent=4)
