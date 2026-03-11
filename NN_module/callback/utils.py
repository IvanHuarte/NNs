import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["toolbar"] = "None"  # ← DESACTIVA icono SVG corrupto
matplotlib.rcParams["figure.raise_window"] = False
import warnings

warnings.filterwarnings("ignore", message="qt.svg")


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


def dump_callback(logger, settings, write=False):

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

    if hasattr(logger, "E_ED"):
        E_gr = np.array(logger.E_ED).real
        error = np.abs(E_hist - E_gr) / np.abs(E_gr)

    # Calculate the variance score
    var = np.real(np.array(logger["Energy"]["Variance"]))
    vscore = N * var / (E_hist**2)
    # vs_min = np.round(np.log10(np.min(vscore)))-1

    setup_sim = f"E_best: {E_best:.4f}\ntime_exe: {time_exe:.2f}"
    if hasattr(logger, "E_ED"):
        setup_sim = f"E_ED: {E_gr:.4f}\n" + setup_sim

    # Plot
    if hasattr(logger, "E_ED"):
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

    if hasattr(logger, "E_ED"):
        ax[0].errorbar(
            range(total_epochs),
            E_hist,
            yerr=dev_E_hist,
            fmt="none",
            ecolor="r",
            label="E_stdev",
        )
        ax[0].hlines(E_gr, 0, total_epochs, color="green", label="ED Energy")

    ax[0].plot(E_hist, color="blue", label="E")
    ax[0].plot(best_step, E_hist[best_step], marker="o", ms=4, color="gold")
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
    ax[v].plot(best_step, vscore[best_step], marker="o", ms=3, color="gold")
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

    if hasattr(logger, "E_ED"):
        ax[e].plot(error, color="red", label="E")
        ax[e].plot(best_step, error[best_step], marker="o", ms=3, color="gold")

        if do_each_checkpoint is not None :
            ax[e].plot(
                checkpoint_indices,
                error[checkpoint_indices],
                ls="",
                marker="o",
                ms=3,
                color="tan",
            )
            
    if hasattr(logger, "E_ED"):
        ax[e].set_yscale("log")
        ax[e].legend()
        ax[e].set_xlabel("Iteration")
        ax[e].set_ylabel("Error", fontsize=12)
        ax[e].grid()

    file_path = write_folder + f"Callback_" + sim_label
    figure_path = file_path + ".jpeg"

    plt.tight_layout()
    plt.savefig(figure_path, dpi=600)
    plt.close()

    callback_artifacts["plot"] = figure_path

    return callback_artifacts
