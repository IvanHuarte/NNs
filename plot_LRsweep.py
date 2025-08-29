#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import os
import argparse
import re
import glob
import json
from pathlib import Path

fontsize_txt=12
fontsize_title=20
fontsize_labels=15
fontsize_legend=13

# Script que desde un directorio padre toma los resultados de cada simulacion de LR
# y hace un plot para cada `alpha` donde se muestra la magnetización, fluctuaciones,
# entropia, y fidelidad para un recorrido en valores de J. La estructura tipica del
# directorio padre es parent_folder/Size_L1xL2/sweeps_x/datos.loquesea donde en el 
# ultimo directorio estan datos de 1 o varios valores de alpha

parser = argparse.ArgumentParser()
parser.add_argument("-d",'--directory', type=str, required=True, help='Directorio padre')
args=parser.parse_args() 
parent=args.directory

parent_folder = parent if parent.endswith("/") else parent + "/"
# Extraemos todos los casos de sweeps
all_subdirs = []
for root, dirs, files in os.walk(parent_folder):
    for d in dirs:
        all_subdirs.append(os.path.join(root, d))

sweep_cleaning = set([os.path.basename(folder) for folder in all_subdirs])
sweep_cases = [name for name in sweep_cleaning if 'sweeps_' in name ]

tree_files = dict((k ,{}) for k in sweep_cases)
for sweep in sweep_cases:
    sizes = glob.glob(parent_folder + f"**/{sweep}")
    for size in sizes:
        # pos = size.find("Size_")
        # size_name = size[pos +len("Size_"):pos +len("Size_") + 3]
        size_name = re.search(r"Size_([0-9]+x[0-9]+)",size).group(1)
        json_objects = glob.glob(size +'/*.json')
        alphas = set([re.search(r"alpha_([\d\.]+)", file).group(0) for file in json_objects])

        for alpha in alphas:
            alpha_files = [file for file in json_objects if alpha in file]
            if not alpha in tree_files[sweep].keys(): tree_files[sweep][alpha]={}
            tree_files[sweep][alpha][size_name] = alpha_files

model_name = os.path.basename(parent).replace("_", "-")

for sweep, sweep_dict in tree_files.items():

    for alpha, alpha_dict in sweep_dict.items():

        fig1, ax1 = plt.subplots(3,1,figsize=[12,20])
        fig2, ax2 = plt.subplots(3,1,figsize=[12,20])
        fig3, ax3 = plt.subplots(2,1,figsize=[8,10])

        for size, files in alpha_dict.items():
            
            results =[[] for _ in range(10)] # For [J, E_best, E_ED, error, vscore, S_renyi, M, Ms, fidelity, timexe]

            for file in files:
                # Load artifacts and simulation params
                with open(file,'r') as f:
                    art = json.load(f)
                # print(art['coupling_model']['J'])
                results[0].append(art['coupling_model']['J'])
                results[1].append(art['results']['E_best'])
                results[2].append(art['results']['E_ED'])
                results[3].append(art['results']['error'])
                results[4].append(art['results']['vscore'])
                results[5].append(art['results']['S_renyi'])
                results[6].append(art['results']['M'])
                results[7].append(art['results']['Ms'])
                results[8].append(art['results']['fidelity'])
                results[9].append(art['results']['time_exe'])

            alpha = art['coupling_model']['alpha']

            results = np.array(results)
            idx = np.argsort(results[0])
            res = results[:,idx]

            # Plot 1: E_best/E_ED, error, vscore
            ax1[0].set_title(r"$%s$"%(model_name) + r"        $\alpha=%.2f$ "%(alpha), fontsize=fontsize_title)
            ax1[0].set_ylabel(r"$Energy$", fontsize=fontsize_labels)
            ax1[0].plot(res[0], res[1], color='blue', alpha=0.8, marker='o', ms=3, lw=1.5, label=r"$E\;(%s)$"%(size))
            ax1[0].plot(res[0], res[2], color='lime',ls='--', marker='o', ms=3, lw=1.5, alpha=0.8, label=r"$E_{ED}\; (%s)$"%(size))
            ax1[0].grid()
            ax1[0].legend(fontsize=fontsize_legend)

            y_min=min(res[3]/2)
            ax1[1].set_ylim(y_min,1)
            ax1[1].set_ylabel(r"$rel.\;\;Error$",fontsize=fontsize_labels)
            ax1[1].plot(res[0], res[3], color='red', alpha=0.8, marker='o', ms=3, lw=1.5,  label=r"$%s$"%(size))
            ax1[1].set_yscale('log')
            ax1[1].grid()
            ax1[1].legend(fontsize=fontsize_legend)
            
            y_min=min(res[4]/2)
            ax1[2].set_ylim(y_min,1)
            ax1[2].set_ylabel(r"$Vscore$", fontsize=fontsize_labels)
            ax1[2].set_xlabel(r"$J$", fontsize=fontsize_labels)
            ax1[2].plot(res[0], res[4], color='purple',marker='o', ms=3 , lw=1.5, alpha=0.8,   label=r"$%s$"%(size))
            ax1[2].set_yscale('log')
            ax1[2].grid()
            ax1[2].legend(fontsize=fontsize_legend)

            # Plot 2: Renyi-entropy,  Magnetization and fluctuation
            ax2[0].set_title(r"$%s$"%(model_name) + r"        $\alpha=%.2f$ "%(alpha), fontsize=fontsize_title)
            ax2[0].set_ylabel(r"$S_{renyi}$", fontsize=fontsize_labels)
            ax2[0].plot(res[0], res[5], color='green', alpha=0.8, marker='o', ms=3, lw=1.5, label=r"$%s$"%(size))
            ax2[0].grid()
            ax2[0].legend(fontsize=fontsize_legend)

            ax2[1].set_ylim(-1,1)
            ax2[1].set_ylabel(r"$Magnetization\;\;(M_z)$",fontsize=fontsize_labels)
            ax2[1].plot(res[0], res[6], color='red', alpha=0.8, marker='o', ms=3, lw=1.5,  label=r"$%s$"%(size))
            ax2[1].grid()
            ax2[1].legend(fontsize=fontsize_legend)

            ax2[2].set_ylim(0,1)
            ax2[2].set_ylabel(r"$Fluctuations\;\;(M_s)$", fontsize=fontsize_labels)
            ax2[2].set_xlabel(r"$J$", fontsize=fontsize_labels)
            ax2[2].plot(res[0], res[7], color='purple',marker='o', ms=3 , lw=1.5, alpha=0.8,   label=r"$%s$"%(size))
            ax2[2].grid()
            ax2[2].legend(fontsize=fontsize_legend)

            # Plot 3: Fidelity and Time Execution
            ax3[0].set_ylim(0.5, 1.05)
            ax3[0].set_title(r"$%s$"%(model_name) + r"        $\alpha=%.2f$ "%(alpha), fontsize=fontsize_title*2/3)
            ax3[0].set_ylabel(r"$Fidelity$", fontsize=fontsize_labels*2/3)
            ax3[0].plot(res[0], res[8], color='darkorange', alpha=0.8, marker='o', ms=3, lw=1.5, label=r"$%s$"%(size))
            ax3[0].grid()
            ax3[0].legend(fontsize=fontsize_legend*2/3)

            ax3[1].set_ylabel(r"$TimeExe\;('')$",fontsize=fontsize_labels*2/3)
            ax3[1].set_xlabel(r"$J$", fontsize=fontsize_labels*2/3)
            ax3[1].plot(res[0], res[9], color='olivedrab', alpha=0.8, marker='o', ms=3, lw=1.5,  label=r"$%s$"%(size))
            ax3[1].grid()
            ax3[1].legend(fontsize=fontsize_legend*2/3)
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()

        fig1.savefig(parent_folder + f"ResultsOfSweep_alpha_{alpha}_Energy_Error_Vscore" + ".jpeg", dpi=600, bbox_inches="tight")
        fig2.savefig(parent_folder + f"ResultsOfSweep_alpha_{alpha}_Renyi_Mz_Ms" + ".jpeg", dpi=600, bbox_inches="tight")
        fig3.savefig(parent_folder + f"ResultsOfSweep_alpha_{alpha}_Fidelity_Timexe" + ".jpeg", dpi=600, bbox_inches="tight")

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
            



