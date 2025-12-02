import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["toolbar"] = "None"  # ← DESACTIVA icono SVG corrupto
matplotlib.rcParams["figure.raise_window"] = False
import warnings

warnings.filterwarnings("ignore", message="qt.svg")


def plot_training_setup(ax, setup):
    segments = []
    modes = []
    lr_schedule = []
    s, m, r, lr_sch = (
        setup["segments"],
        setup["mode"],
        setup["repeat_segment"],
        setup["lr"],
    )

    for seg, mode, repeats, lrsch in zip(s, m, r, lr_sch):
        segments += [seg] * repeats
        modes += [mode] * repeats
        lr_schedule += [lrsch] * repeats

    total_epochs = int(np.array([s for seg in segments for s in seg]).sum())

    cum_epoch = 0
    for i, (segment, seg_modes, lr) in enumerate(zip(segments, modes, lr_schedule)):

        if i != 0:
            ax.axvline(cum_epoch, color="grey", linestyle="--", alpha=0.5)

        ax.text(
            cum_epoch + segment[0] / 2,
            0.7,
            f"{''.join(seg_modes)}\n{lr}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            color="black",
            transform=ax.get_xaxis_transform(),
        )

        cum_epoch += segment[0]


def dump_callback(logger, settings, write=False):

    callback_artifacts = {}

    time_exe = settings["time_exe"]
    write_folder = settings["write_folder"]
    sim_label = settings["sim_label"]
    title_label_callback = settings["title_label_callback"]
    best_step = settings["best_step"]
    training_setup = settings["training_setup"]

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

    setup_sim = f"E_best: {E_best:.4f}\nlr:{training_setup['lr_name']} \ntime_exe: {time_exe:.2f}"
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

    ax[0].set_title(title_label_callback)

    if hasattr(logger, "E_ED"):
        ax[0].errorbar(
            range(len(E_hist)),
            E_hist,
            yerr=dev_E_hist,
            fmt="none",
            ecolor="r",
            label="E_stdev",
        )
        ax[0].hlines(E_gr, 0, len(E_hist), color="green", label="ED Energy")

    ax[0].plot(E_hist, color="blue", label="E")
    ax[0].plot(best_step, E_hist[best_step], marker="o", ms=3, color="gold")

    ax[0].text(
        0.9,
        0.75,
        setup_sim,
        transform=ax[0].transAxes,
        fontsize=10,
        color="k",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    # Plotting training setup
    if training_setup["lr_name"] == "segment":
        plot_training_setup(ax[0], training_setup)

    ax[0].legend()
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Energy", fontsize=12)
    ax[0].grid()

    ax[v].plot(vscore, color="purple", label="Vscore")
    ax[v].plot(best_step, vscore[best_step], marker="o", ms=3, color="gold")
    ax[v].set_yscale("log")
    # ax[v].set_ylim(bottom=vs_min)
    ax[v].legend()
    ax[v].set_xlabel("Iteration")
    ax[v].set_ylabel("Vscore", fontsize=12)
    ax[v].grid()

    if hasattr(logger, "E_ED"):
        ax[e].plot(error, color="red", label="E")
        ax[e].plot(best_step, error[best_step], marker="o", ms=3, color="gold")
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

    # Save the data
    if write:
        E_path = file_path + "_E_hist.txt"
        error_path = file_path + "_error.txt"
        vscore_path = file_path + "_vscore.txt"

        np.savetxt(E_path, np.array(E_hist))
        np.savetxt(error_path, np.array(error))
        np.savetxt(vscore_path, np.array(vscore))

        callback_artifacts["Energy"] = E_path
        callback_artifacts["error"] = error_path
        callback_artifacts["vscore"] = vscore_path

    return callback_artifacts
