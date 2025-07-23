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

from NN_module.sim_utils import load_vstate
from NN_module.NN_utils import modphase_extended
from NN_module.correlations import correlations_ED, correlations_vstate

# Añadir el directorio chebyoxa al path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA"))

parser = argparse.ArgumentParser()
parser.add_argument("-a",'--artifact_path', type=str, required=True, help='Path al artefacto principal que recoge los resultados de la simulacion')
parser.add_argument("-w",'--write_folder', type=str, help='Path donde escribir los SSF')
parser.add_argument("-exp", '--explore_mode', type=str, help='Modo para discriminar simulaciones con los mismos parametros')
args=parser.parse_args()

path_artifact= args.artifact_path
write_folder = args.write_folder if args.write_folder is not None else os.path.dirname(path_artifact) + "/"
explore_mode = args.explore_mode

# Load artifact and simulation params
with open(path_artifact,'r') as f:
    artifact = json.load(f)

strength = artifact['coupling_model']['strength']
size = artifact['lattice']['size']
theta = artifact['coupling_model']['theta']
phi = artifact['coupling_model']['phi']

filename = f"Oxalate_" + artifact["model_NN"]["name"] + f"_SSF_strength_{strength:2f}_theta_{theta:2f}_phi_{phi:2f}"

# Verify there is no previous simulations, add an int label otherwise.
print(f"\n************size: {size} theta: {theta:1f}  phi: {phi:1f} ************\n\n")
if explore_mode:                    # If True checks sucessive files and assign a new one
    if os.path.isfile(write_folder + filename + ".jpeg"):
        i=1
        file_temp = filename + f"_{i}.txt"
        while(os.path.isfile(write_folder + file_temp)):
            file_temp = filename + f"_{i}.txt"
            i+=1
            print(f"_{i}.txt")
        
        filename += f"_{i}.txt"


ED_file=None
if 'x_ED' in artifact['_artifacts']:
    ED_file = artifact['_artifacts']['x_ED']

# Get modulus and phase
vstate = load_vstate(artifact, tree_data=False)
x_vs = vstate.to_array()
mod_vs, phase_vs, stats_vs = modphase_extended(x_vs)

if ED_file is not None:
    x_ED=np.loadtxt(ED_file, dtype=complex)
    mod_ED, phase_ED, stats_ED = modphase_extended(x_ED)



# Plot
# If ED exists
if ED_file is not None:
        
    _,ax= plt.subplots(4,1, figsize=[15,10])

    ax[0].set_title(r"$Modulus\;and\;Phase\qquad Oxalate\;size\;%d x %d \qquad a=%.1f\;\;\theta=%.1f \;\; \phi=%.1f$"%(size[0],size[1],strength,theta,phi))
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"$Modulus$")
    ax[0].set_ylim(-0.01,max(max(mod_ED),max(mod_vs))*9/8)
    ax[0].plot(mod_vs, alpha=0.6, color='r', label='vstate')
    ax[0].plot(mod_ED, alpha=0.6, label='ED')
    ax[0].legend()

    ax[1].set_xticks([])
    ax[1].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax[1].set_yticklabels([r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
    ax[1].set_ylabel(r"$Phase \;vstate$")
    ax[1].set_ylim(-np.pi-0.1, np.pi+0.1)
    ax[1].plot(phase_vs, alpha=0.25, ls='', marker='o', ms=0.9, color='r', label='vstate')
    ax[1].legend(loc='upper right')

    ax[2].set_xlabel(r"$C_i$")
    ax[2].set_ylabel(r"$Phase \;ED$")
    ax[2].set_ylim(-np.pi-0.1, np.pi+0.1)
    ax[2].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax[2].set_yticklabels([r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
    ax[2].plot(phase_ED, alpha=0.25, ls='', marker='o', ms=0.9, label='ED')
    ax[2].legend(loc='upper right')

    ax[3].set_xlabel(r"$Phase\;(radians)$")
    ax[3].set_ylabel(r"$Phase \;histogram$")
    ax[3].hist(phase_ED, bins=1000, range=(-np.pi, np.pi), density=True, alpha=0.7, label='ED')
    ax[3].hist(phase_vs, bins=1000, range=(-np.pi, np.pi), color='r', density=True, alpha=0.7, label='vstate')


    transform = mtransforms.blended_transform_factory(ax[3].transData, ax[3].transAxes)
    if stats_ED['peaks'] is not None:
        for peak in stats_ED['peaks']['values']:
            ax[3].text(peak-0.1, 0.9, r"%.2f"%peak, color='b', transform=transform, fontsize=8, alpha=0.7)

    if stats_vs['peaks'] is not None:
        for peak in stats_vs['peaks']['values']:
            ax[3].text(peak-0.1, 0.9, r"%.2f"%peak, color='b', transform=transform, fontsize=8, alpha=0.7)
    ax[3].legend()
    plt.tight_layout()
    plt.savefig(write_folder + filename + ".jpeg", dpi=600, bbox_inches="tight")

else:

    _, ax= plt.subplots(3,1, figsize=[13,9])

    ax[0].set_title(r"$Modulus\;and\;Phase\qquad Oxalate\;size\;%d x %d \qquad a=%.1f\;\;\theta=%.1f \;\; \phi=%.1f$"%(size[0],size[1],strength,theta,phi))
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"$Modulus$")
    ax[0].set_ylim(-0.01,max(max(mod_ED),max(mod_vs))*9/8)
    ax[0].plot(mod_vs, alpha=0.6, color='r', label='vstate')
    ax[0].plot(mod_ED, alpha=0.6, label='ED')
    ax[0].legend()

    ax[2].set_xlabel(r"$C_i$")
    ax[1].set_xticks([])
    ax[1].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax[1].set_yticklabels([r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
    ax[1].set_ylabel(r"$Phase \;vstate$")
    ax[1].set_ylim(-np.pi-0.1, np.pi+0.1)
    ax[1].plot(phase_vs, alpha=0.25, ls='', marker='o', ms=0.9, color='r', label='vstate')
    ax[1].legend(loc='upper right')


    ax[2].set_xlabel(r"$Phase\;(radians)$")
    ax[2].set_ylabel(r"$Phase \;histogram$")
    ax[2].hist(phase_vs, bins=1000, range=(-np.pi, np.pi), color='r', density=True, alpha=0.7, label='vstate')


    transform = mtransforms.blended_transform_factory(ax[3].transData, ax[3].transAxes)

    if stats_vs['peaks'] is not None:
        for peak in stats_vs['peaks']['values']:
            ax[3].text(peak-0.1, 0.9, r"%.2f"%peak, color='b', transform=transform, fontsize=8, alpha=0.7)
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(write_folder + filename + ".jpeg", dpi=600, bbox_inches="tight")


artifact['_artifacts']['modphase_plot'] = write_folder + filename + ".jpeg"
with open(path_artifact,"w") as f:
    json.dump(artifact, f, separators=(",", ":"), sort_keys=True, indent=4)