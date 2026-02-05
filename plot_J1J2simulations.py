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
import shutil


fontsize_txt = 12
fontsize_ticks = 15
fontsize_title = 20
fontsize_labels = 15
fontsize_legend = 13

# Script que desde un directorio padre toma los resultados de cada simulacion de J1J2
# y hace un plot para cada simulacion donde se muestra la magnetización, fluctuaciones,
# entropia, y fidelidad para un recorrido en valores de J.

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--directory", type=str, required=True, help="Directorio padre"
)
args = parser.parse_args()
parent = args.directory

parent_folder = parent if parent.endswith("/") else parent + "/"
# Extraemos todos los casos de sweeps

write = "/home/ihuarte/Escritorio/Ivan/NNs/Results/"

write = write if write.endswith("/") else write + "/"
target = parent.split(str(Path(parent).parent) + "/")[1]
write += target if target.endswith("/") else target + "/"
parent_folder = parent if parent.endswith("/") else parent + "/"


fontsize_txt = 12
fontsize_ticks = 15
fontsize_title = 20
fontsize_labels = 15
fontsize_legend = 13


# Extraemos todos los casos de sweeps
all_subdirs = []
for root, dirs, files in os.walk(parent_folder):
    # print(dirs)
    for d in dirs:
        if not d.endswith(".orbax") and ".orbax" not in root:
            all_subdirs.append(os.path.join(root, d))

all_subdirs = sorted(all_subdirs)
all_subdirs = [d + "/" for d in all_subdirs if "UUID_" in d]
parent_folder_subdirs = [d.split(parent_folder)[1] for d in all_subdirs]

for subdir in parent_folder_subdirs:                                                #### BUCLE DE SIMULACIONES
    artifact_paths = glob.glob(parent_folder + subdir + "*_results_*.json")
    J12_vals_path = []
    for d in artifact_paths:
        J1, J2 = re.search(r"_J1J2_([0-9\.]+_[0-9\.]+)_", d).group(1).split("_")
        J12_vals_path.append([float(J1), float(J2), d])

    sorted_J12 = sorted(J12_vals_path, key=lambda x: x[1]/x[0])

    fig1, ax1 = plt.subplots(3, 1, figsize=[12, 20])
    fig2, ax2 = plt.subplots(3, 1, figsize=[12, 20])
    fig3, ax3 = plt.subplots(3, 1, figsize=[12, 20])

    results = [
        [] for _ in range(16)
    ]

    for J1, J2, art_path in sorted_J12:

        write_folder = write + subdir
        Path(write_folder).mkdir(parents=True, exist_ok=True)

        # Load artifacts and simulation params
        with open(art_path, "r") as f:
            art = json.load(f)
        
        size = art['CM']['size']
        size = f"{size[0]}x{size[1]}"

        model_name = art['NN']['name']
            
        # print(art['coupling_model']['J'])
        results[0].append(J2 / J1)
        results[1].append(art["results"]["E_best"])
        results[2].append(art["results"]["E_ED"])
        results[3].append(art["results"]["error"])
        results[4].append(art["results"]["vscore"])
        results[5].append(art["results"]["S_renyi"])

        results[6].append(art["results"]["m"])
        results[7].append(art["results"]["ms"])
        results[8].append(art["results"]["m2"])
        results[9].append(art["results"]["ms2"])

        results[10].append(art["results"]["m_ED"])
        results[11].append(art["results"]["ms_ED"])
        results[12].append(art["results"]["m2_ED"])
        results[13].append(art["results"]["ms2_ED"])

        results[14].append(art["results"]["fidelity"])
        results[15].append(art["results"]["time_exe"])

        if J2 / J1 == 0.5:
            callback_path = art["_artifacts"]["callback"]["plot"]
            shutil.copy2(callback_path, write_folder)
    
    results = np.array(results)

    (
        J21,
        E_best,
        E_ED,
        error,
        vscore,
        S_renyi,
        m,
        ms,
        m2,
        ms2,
        m_ED,
        ms_ED,
        m2_ED,
        ms2_ED,
        fidelity,
        timexe,
    ) = results

    m_phase = np.where(J21 < 0, m, ms)
    m_phase_ED = np.where(J21 < 0, m_ED, ms_ED)
    m2_phase = np.where(J21 < 0, m2, ms2)
    m2_phase_ED = np.where(J21 < 0, m2_ED, ms2_ED)

    # Plot 1: E_best/E_ED, error, vscore
    ax1[0].set_title(
        r"$%s \qquad J_1=%.2f \quad (%s)$"
        % (model_name, J1, size),
        fontsize=fontsize_title,
    )
    ax1[0].set_ylabel(r"$Energy$", fontsize=fontsize_labels)
    ax1[0].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1[0].plot(
        J21,
        E_best,
        color="blue",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=r"$E\;(%s)$" % (size),
    )
    ax1[0].plot(
        J21,
        E_ED,
        color="lime",
        ls="--",
        marker="o",
        ms=3,
        lw=1.5,
        alpha=0.8,
        label=r"$E_{ED}\; (%s)$" % (size),
    )
    ax1[0].grid()
    ax1[0].legend(fontsize=fontsize_legend)

    y_min = min(error / 2)
    ax1[1].set_ylim(y_min, 1)
    ax1[1].set_ylabel(r"$rel.\;\;Error$", fontsize=fontsize_labels)
    ax1[1].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1[1].plot(
        J21,
        error,
        color="red",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=r"$%s$" % (size),
    )
    ax1[1].set_yscale("log")
    ax1[1].grid()
    ax1[1].legend(fontsize=fontsize_legend)

    y_min = min(vscore / 2)
    ax1[2].set_ylim(y_min, 1)
    ax1[2].set_ylabel(r"$Vscore$", fontsize=fontsize_labels)
    ax1[2].set_xlabel(r"$J_2 / J_1$", fontsize=fontsize_labels)
    ax1[2].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1[2].plot(
        J21,
        vscore,
        color="purple",
        marker="o",
        ms=3,
        lw=1.5,
        alpha=0.8,
        label=r"$%s$" % (size),
    )
    ax1[2].set_yscale("log")
    ax1[2].grid()
    ax1[2].legend(fontsize=fontsize_legend)

    # Plot 2: Renyi-entropy,  Magnetization and fluctuation
    ax2[0].set_title(
        r"$%s \qquad J_1=%.2f \quad (%s)$"
        % (model_name, J1, size),
        fontsize=fontsize_title,
    )    
    ax2[0].set_ylabel(r"$S_{renyi}$", fontsize=fontsize_labels)
    ax2[0].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax2[0].plot(
        J21,
        S_renyi,
        color="green",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=r"$%s$" % (size),
    )
    ax2[0].grid()
    ax2[0].legend(fontsize=fontsize_legend)

    ax2[1].set_ylim(-1, 1)
    ax2[1].set_ylabel(r"$Magnetization\;\;(M_z)$", fontsize=fontsize_labels)
    ax2[1].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax2[1].plot(
        J21,
        m_phase,
        color="red",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=r"$%s$" % (size),
    )
    ax2[1].plot(
        J21,
        m_phase_ED,
        color="lime",
        ls="--",
        alpha=0.8,
        lw=1.5,
        label=r"$%s\;(ED)$" % (size),
    )
    ax2[1].grid()
    ax2[1].legend(fontsize=fontsize_legend)

    ax2[2].set_ylim(0, 1)
    ax2[2].set_ylabel(r"$Fluctuations\;\;(M_s)$", fontsize=fontsize_labels)
    ax2[2].set_xlabel(r"$J_2 / J_1$", fontsize=fontsize_labels)
    ax2[2].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax2[2].plot(
        J21,
        m2_phase,
        color="purple",
        marker="o",
        ms=3,
        lw=1.5,
        alpha=0.8,
        label=r"$%s$" % (size),
    )
    ax2[2].plot(
        J21,
        m2_phase_ED,
        color="lime",
        ls="--",
        lw=1.5,
        alpha=0.8,
        label=r"$%s\;(ED)$" % (size),
    )
    ax2[2].grid()
    ax2[2].legend(fontsize=fontsize_legend)


    # Plot 3: Fidelity and Time Execution
    ax3[0].set_ylim(0.5, 1.05)
    ax3[0].set_title(
        r"$%s \qquad J_1=%.2f \quad (%s)$"
        % (model_name, J1, size),
        fontsize=fontsize_title * 2 / 3,
    )
    ax3[0].set_ylabel(r"$Fidelity$", fontsize=fontsize_labels * 2 / 3)
    ax3[0].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax3[0].plot(
        J21,
        fidelity,
        color="darkorange",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=r"$%s$" % (size),
    )
    ax3[0].grid()
    ax3[0].legend(fontsize=fontsize_legend * 2 / 3)

    ax3[1].set_ylabel(r"$TimeExe\;('')$", fontsize=fontsize_labels * 2 / 3)
    ax3[1].set_xlabel(r"$J_2 / J_1$", fontsize=fontsize_labels * 2 / 3)
    ax3[1].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax3[1].plot(
        J21,
        timexe,
        color="olivedrab",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=r"$%s$" % (size),
    )
    ax3[1].grid()
    ax3[1].legend(fontsize=fontsize_legend * 2 / 3)

    from NN_module.schedule.utils import calculate_lr_schedule
    lr, info  = calculate_lr_schedule(art['SIM']['schedule']['learning_rate'])
    lr_concat = np.concatenate(lr, axis=-1)
    total_epochs = art['SIM']['schedule']['total_epochs']
    
    ax3[2].set_xlabel(r"$Epochs$", fontsize=fontsize_labels)
    ax3[2].set_ylabel(r"$Learning\;Rate$", fontsize=fontsize_labels)
    ax3[2].plot(range(total_epochs),lr_concat)
    ax3[2].set_yscale("log")
    ax3[2].grid(which="both", ls="--")

    cum = 0
    for inf in info:
        x_pos = (cum + inf[0])/2
        
        label = f"{inf[2]}\n" + f"{inf[1]}"
        ax3[2].axvline(x_pos*2, color="grey", linestyle="--")
        ax3[2].text(
            x_pos,
            0.95,
            label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color="black",
            transform=ax3[2].get_xaxis_transform(),
        )
        cum += inf[0]


    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig(
        write_folder
        + f"RunInJ_J1_{J1}_Energy_Error_Vscore"
        + ".jpeg",
        dpi=600,
        bbox_inches="tight",
    )
    fig2.savefig(
        write_folder
        + f"RunInJ_J1_{J1}_Renyi_Mz_Ms"
        + ".jpeg",
        dpi=600,
        bbox_inches="tight",
    )
    fig3.savefig(
        write_folder
        + f"RunInJ_J1_{J1}_Fidelity_Timexe_lr"
        + ".jpeg",
        dpi=600,
        bbox_inches="tight",
    )
    

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


