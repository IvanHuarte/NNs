#!/home/ihuarte/miniconda3/envs/conda_env/bin/python3

import os
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import patches
import netket as nk
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import json
import seaborn as sns
import sys
from pathlib import Path

# Añadir el directorio chebyoxa
sys.path.append(str(Path(__file__).resolve().parent.parent / "chebyoxa"))

import chebyoxa.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--artifact_path', type=str, required=True, help='Path al artefacto principal que recoge los resultados de la simulacion')
parser.add_argument('--write_folder', type=str, required=True, help='Path donde guardar los plots de los SSF')
parser.add_argument('--explore_mode', type=bool, help='Modo para discriminar simulaciones con los mismos parametros', default=False)
parser.add_argument('--plot_error', type=bool, help='Plotear el error relativo entre el SSF obtenido y el de ED', default=False)
args=parser.parse_args()

path_artifact= args.artifact_path
write_folder = args.write_folder
explore_mode = args.explore_mode
plot_error = args.plot_error

with open(path_artifact,'r') as f:
    artifact = json.load(f)

strength = artifact['coupling_model']['strength']
size = artifact['lattice']['size']
theta = artifact['coupling_model']['theta']
phi = artifact['coupling_model']['phi']

ssf_path = artifact['_artifacts']['SSF']['OPT']
ssf_ED_path = artifact['_artifacts']['SSF']['ED']
write_folder_fig = write_folder + f"Oxalate_size_{size[0]}x{size[1]}/"

file = f"Structure_Factor_size{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_{artifact['model_NN']['name']}"
file_ED = f"Structure_Factor_size{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_ED"
file_error = f"Structure_Factor_size{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_error"
files = [file]

# Check exact diagonalization mode
SSF_label=[artifact["model_NN"]["name"]]
if artifact["results"]["E_ED"] is not None:
    exact_diag = True
else:
    exact_diag = False

if os.path.isfile(write_folder_fig + file_ED):
    exact_diag = False

if exact_diag:
    SSF_label.append('ED')
    files.append(file_ED)

#Some Sebas's stuff
A = strength
N_Q_A = 48 * 3
N_Q_B = 48 * 3


if not os.path.exists(write_folder_fig):
    os.makedirs(write_folder_fig)

graph = nk.graph.Triangular(size, pbc=True)
qs_mapping, direct_qs = utils.get_q_mesh(N_Q_A, N_Q_B, graph)

if not os.path.isfile(ssf_path):
    exit(1, f"File {ssf_path} not found")

# Verify there is no previous simulations, to earn time

if explore_mode:                    # If True checks sucessive files and assign a new one
    file_temp = file + ".jpeg"
    i=0
    while(os.path.isfile(write_folder_fig + file_temp)):
        i+=1
        file_temp = file + f"_{i}.jpeg"
    
    files[0] += f"_{i}"

else:
    if os.path.isfile(write_folder_fig + file): 
        exit(0, f"File {write_folder_fig + file} already exists")

# Plotting the SSF
SSF = np.loadtxt(ssf_path)

plots_ssf = [SSF]
E_best = artifact["results"]["E_best"]
label=[]

E_ED = artifact["results"]["E_ED"]
if exact_diag:

    SSF_ED = np.loadtxt(ssf_ED_path)
    plots_ssf.append(SSF_ED)

    if plot_error:
        error = artifact["results"]["error"]
        SSF_label.append('Error')  
        SSF_error = np.abs(SSF - SSF_ED) / np.abs(SSF_ED)
        plots_ssf.append(SSF_error)
        files.append(file_error)

cmap = sns.color_palette("mako", as_cmap=True)

model_data=""
for i, data in enumerate(artifact["model_NN"].values()):
    model_data += str(data)
    if i < len(artifact["model_NN"].values()) - 1:
        model_data += "\n"


for j, SSF in enumerate(plots_ssf):
    if j == 2:
        cmap = sns.color_palette("inferno", as_cmap=True)

    tiled_structure_factors = np.tile(SSF, (3, 3))
    reciprocal_vectors = utils.get_reciprocal_vectors(graph)
    bz_vertices = utils.get_brillouin_zone(reciprocal_vectors)
    bz_patch = patches.Polygon(bz_vertices, edgecolor="red", fill=False)

    plt.figure(j)
    ax = plt.gca()
    im = plt.imshow(
        tiled_structure_factors,
        origin="lower",
        extent=(-1.5, 1.5, -1.5, 1.5),
        cmap=cmap,
        interpolation="none",
    )

    affine_matrix = utils.get_affine_transformation(reciprocal_vectors)
    transform = mtransforms.Affine2D()
    transform.set_matrix(affine_matrix.T)
    trans_data = transform + ax.transData

    im.set_transform(trans_data)
    bz_patch.set_transform(trans_data)

    ax.add_patch(bz_patch)
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--", transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4.2, 4.2)
    plt.colorbar(im, ax=ax)
    plt.xlabel(r"$k_x/\pi$")
    plt.ylabel(r"$k_y/\pi$")
    plt.title(r"$a= %.1f$  $\theta = %.1f$  $\phi = %.1f$    "%(strength,theta,phi)+ SSF_label[j]+ f"   {size}")

    if j == 0:
        
        if E_ED is None:
            ax.text(-0.1, -0.18, f"                 OPT\nEnergy: {E_best:3.4f} \nE/N:    {E_best/np.prod(size):3.4f}",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor="white", alpha=0.4))
        else:
            ax.text(-0.1, -0.18, f"                 OPT            ED            error \nEnergy: {E_best:3.5f}   {E_ED:3.5f}   {error:3.3e} \nE/N:      {E_best/np.prod(size):3.5f}   {E_ED/np.prod(size):3.5f}   {error/np.prod(size):.3e}",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor="white", alpha=0.4))
        ax.text(0.90, -0.18, f"{model_data}", transform=ax.transAxes, fontsize=8, bbox=dict(facecolor="white", alpha=0.4))

    if j == 1:
        ax.text(-0.1, -0.18, f"                 ED\nEnergy: {E_ED:3.4f} \nE/N:      {E_ED/np.prod(size):3.4f}",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor="white", alpha=0.4))

    if j == 2:
        ax.text(-0.1, -0.18, f"                  OPT           ED           error \nEnergy: {E_best:3.5f}  {E_ED:3.5f}  {error:3.3e} \nE/N:      {E_best/np.prod(size):.5f}   {E_ED/np.prod(size):.5f}   {error/np.prod(size):.3e}",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor="white", alpha=0.4))

    plt.tight_layout()
    #plt.show()
    plt.savefig(write_folder_fig + files[j] + ".jpeg", dpi=600, bbox_inches="tight")
    plt.close(j)