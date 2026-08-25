import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["toolbar"] = "None"  # ← DESACTIVA icono SVG corrupto
matplotlib.rcParams["figure.raise_window"] = False
import warnings

warnings.filterwarnings("ignore", message="qt.svg")

from NN_module.label_utils import get_filenames_from_settings
from NN_module.schedule.schedule import Schedule


def plot_schedule_setup(ax, setup):

    epochs, modes, lr_ins, rescale, eons_bound = (
        setup["epochs"],
        setup["modes"],
        setup["lr_instructions"],
        setup["rescale"],
        setup["eons_bound"],
    )

    total_epochs = int(np.array([epo for epo in epochs]).sum())
    ins_th = total_epochs // 15

    cum_epoch = 0
    for i, (epo, mode, lr) in enumerate(zip(epochs, modes, lr_ins)):

        if i != 0:
            ax.axvline(cum_epoch, color="grey", linestyle="--", alpha=0.5)

        period_label = f"{''.join(mode)}"
        if epo > ins_th:
            for l in lr:
                period_label += f"\n{l}"

        ax.text(
            (cum_epoch + epo) / 2,
            0.85 + 0.05 * (-1) ** i,
            period_label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color="black",
            transform=ax.get_xaxis_transform(),
        )

        cum_epoch += epo

    if rescale != 1.0:
        ax.text(
            0.5,
            0.95,
            f"Rescaled by {rescale}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            color="black",
            transform=ax.transAxes,
        )
    eons_bound = [0] + eons_bound
    for i in range(len(eons_bound[1:])):
        ax.axvline(eons_bound[i], color="black", linestyle="--", alpha=0.7)
        ax.text(
            eons_bound[i] + eons_bound[i + 1] / 2,
            0.95,
            r"$Stage\;%d$" % i,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            color="black",
            transform=ax.get_xaxis_transform(),
        )


def dump_callback(logger, settings, write_callback=True):

    callback_artifacts = {}

    time_exe = settings["time_exe"]
    write_folder = settings["write_folder"]
    sim_label = settings["sim_label"]
    title_label_callback = settings["title_label_callback"]
    best_step = settings["best_step"]
    schedule_setup = settings["schedule_setup"]
    do_each_checkpoint = settings["do_each_checkpoint"]

    N = int(np.prod(settings["size"]))

    os.makedirs(write_folder, exist_ok=True)

    # Extract some results
    E_hist = np.array(logger["Energy"]["Mean"]).real
    dev_E_hist = np.array(logger["Energy"]["Sigma"]).real
    E_best = min(logger["Energy"]["Mean"]).real
    best_color = "gold"  
    if best_step >= len(E_hist):
        best_step = len(E_hist) - 1
        best_color = "red" 

    if hasattr(logger, "E_gr_global"):
        E_gr_global = np.array(logger.E_gr_global).real
        error = np.abs(E_hist - E_gr_global) / np.abs(E_gr_global)
        if hasattr(logger, "E_gr_irrep"):
            irrep = logger.irrep
            symmetries = logger.symmetries
            E_gr_irrep = logger.E_gr_irrep
            error = np.abs(E_hist - E_gr_irrep) / np.abs(E_gr_irrep)


    # Calculate the variance score
    var = np.real(np.array(logger["Energy"]["Variance"]))
    vscore = N * var / (E_hist**2)
    # vs_min = np.round(np.log10(np.min(vscore)))-1

    setup_sim = f"E_best: {E_best:.4f}\ntime_exe: {time_exe:.2f}"
    if hasattr(logger, "E_gr_global"):
        if hasattr(logger, "E_gr_irrep"):
            setup_sim = f"E_gr irrep: {E_gr_irrep:.4f}\n" + setup_sim
        setup_sim = f"E_gr global: {E_gr_global:.4f}\n" + setup_sim

    # Plot
    if hasattr(logger, "E_gr_global"):
        _, ax = plt.subplots(3, 1, figsize=(8, 18))
        v = 2
        e = 1
    else:
        _, ax = plt.subplots(2, 1, figsize=(8, 12))
        v = 1
        e = 2

    total_epochs = len(E_hist)
    if do_each_checkpoint is not None:
        checkpoint_indices = np.arange(0, total_epochs, do_each_checkpoint)

    ax[0].set_title(title_label_callback)

    if hasattr(logger, "E_gr_global"):
        ax[0].errorbar(
            range(total_epochs),
            E_hist,
            yerr=dev_E_hist,
            fmt="none",
            ecolor="r",
            label="E_stdev",
        )
        ax[0].hlines(E_gr_global, 0, total_epochs, color="green", label="ED global")
        if hasattr(logger, "E_gr_irrep"):
            ax[0].hlines(E_gr_irrep, 0, total_epochs, color="orange", ls="--", label=f"ED {irrep}")



    ax[0].plot(E_hist, color="blue", label="E")
    ax[0].plot(best_step, E_hist[best_step], marker="o", ms=4, color=best_color)
    if do_each_checkpoint is not None:
        ax[0].plot(
            checkpoint_indices,
            E_hist[checkpoint_indices],
            ls="",
            marker="o",
            ms=3,
            color="tan",
        )

    ax[0].text(
        0.9,
        0.9,
        setup_sim,
        transform=ax[0].transAxes,
        fontsize=10,
        color="k",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    # Plotting training setup
    plot_schedule_setup(ax[0], schedule_setup) if schedule_setup is not None else None

    ax[0].legend()
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Energy", fontsize=12)
    ax[0].grid()

    ax[v].plot(vscore, color="purple", label="Vscore")
    ax[v].plot(best_step, vscore[best_step], marker="o", ms=3, color=best_color)
    if do_each_checkpoint is not None:
        ax[v].plot(
            checkpoint_indices,
            vscore[checkpoint_indices],
            ls="",
            marker="o",
            ms=3,
            color="tan",
        )

    ax[v].set_yscale("log")
    # ax[v].set_ylim(bottom=vs_min)
    ax[v].legend()
    ax[v].set_xlabel("Iteration")
    ax[v].set_ylabel("Vscore", fontsize=12)
    ax[v].grid()

    if hasattr(logger, "E_gr_global"):
        ax[e].plot(error, color="red", label="E")
        ax[e].plot(best_step, error[best_step], marker="o", ms=3, color=best_color)

        if do_each_checkpoint is not None:
            ax[e].plot(
                checkpoint_indices,
                error[checkpoint_indices],
                ls="",
                marker="o",
                ms=3,
                color="tan",
            )

        ax[e].set_yscale("log")
        ax[e].legend()
        ax[e].set_xlabel("Iteration")
        ax[e].set_ylabel("Error", fontsize=12)
        ax[e].grid()

    exit_message = settings["message"]
    ax[-1].text(0.6, -0.1, exit_message, transform=ax[-1].transAxes)
  
   
    file_path = write_folder + f"Callback_" + sim_label
    figure_path = file_path + ".jpeg"

    plt.tight_layout()
    plt.savefig(figure_path, dpi=600)
    plt.close()

    callback_artifacts["plot"] = figure_path
   

    if write_callback:
        file_path_energy = file_path + "_energy.txt"
        file_path_energy_dev = file_path + "_energy_dev.txt"
        file_path_vscore = file_path + "_vscore.txt"

        np.savetxt(file_path_energy, E_hist)
        np.savetxt(file_path_energy_dev, dev_E_hist)
        np.savetxt(file_path_vscore, vscore)

        callback_artifacts["energy"] = file_path_energy
        callback_artifacts["energy_dev"] = file_path_energy_dev
        callback_artifacts["vscore"] = file_path_vscore


        if hasattr(logger, "E_gr_global"):
            file_path_error = file_path + "_error.txt"
            np.savetxt(file_path_error, error)
            callback_artifacts["error"] = file_path_error

    return callback_artifacts


def checkpoint_callback(artifact, steps, energy, vscore):

    E_gr_global = None


    if artifact is not None:
        time_exe = artifact["results"]["time_exe"]
        best_step = artifact["results"]["best_step"]
        schedule_setup = Schedule(artifact["SIM"]["schedule"], {}).flat_setup()
        total_epochs = artifact["SIM"]["schedule"]["total_epochs"]
        E_best = artifact["results"]["E_best"]
        if "E_gr_global" in artifact["results"]:
            E_gr_global = np.array(artifact["results"]["E_gr_global"]) * 4
            error = np.abs(energy - E_gr_global) / np.abs(E_gr_global)
        _, _, _, callback = get_filenames_from_settings(
            artifact["CM"], {"name": artifact["NN"]["name"], "setup": {}}
        )

    else:
        time_exe = 0.0
        best_step = 0
        schedule_setup = None
        total_epochs = steps[-1]
        E_best = 0.0
        error = None
        callback = "Callback"


    setup_sim = (
        f"E_best: {E_best:.4f}\ntime_exe: {time_exe:.2f}\nBest step: {best_step:d}"
    )

    if E_ED is not None:
        setup_sim = f"E_ED: {E_ED:.4f}\n" + setup_sim

    # Plot
    if E_ED is not None:
        fig, ax = plt.subplots(3, 1, figsize=(8, 18))
        v = 2
        e = 1
    else:
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        v = 1
        e = 2

    ax[0].set_title(callback)

    if E_ED is not None:
        ax[0].hlines(E_ED, 0, total_epochs, color="green", label="ED Energy")

    ax[0].plot(steps, energy, color="blue", marker="o", ms=3, label="E")

    ax[0].text(
        0.9,
        0.9,
        setup_sim,
        transform=ax[0].transAxes,
        fontsize=10,
        color="k",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    # Plotting training setup
    plot_schedule_setup(ax[0], schedule_setup) if schedule_setup is not None else None

    ax[0].legend()
    ax[0].set_xlim(0, 1.01 * total_epochs)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Energy", fontsize=12)
    ax[0].grid()

    ax[v].plot(steps, vscore, color="purple", label="Vscore")
    ax[v].set_yscale("log")
    ax[v].legend()
    ax[v].set_xlim(0, 1.01 * total_epochs)
    ax[v].set_xlabel("Iteration")
    ax[v].set_ylabel("Vscore", fontsize=12)
    ax[v].grid()

    if E_ED is not None:
        ax[e].plot(steps, error, color="red", label="E")
        ax[e].set_yscale("log")
        ax[e].legend()
        ax[e].set_xlim(0, 1.01 * total_epochs)
        ax[e].set_xlabel("Iteration")
        ax[e].set_ylabel("Error", fontsize=12)
        ax[e].grid()

    return fig, ax
