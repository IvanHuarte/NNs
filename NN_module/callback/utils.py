import os
import warnings
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["toolbar"] = "None"  # ← DESACTIVA icono SVG corrupto
matplotlib.rcParams["figure.raise_window"] = False

warnings.filterwarnings("ignore", message="qt.svg")

from NN_module.label_utils import labels_from_artifact

# Older artifacts stored the energies divided by 4 (Pauli -> S units) while
# checkpoint metrics kept the raw Hamiltonian value. New artifacts carry
# results["energy_units"] and need no rescaling.
LEGACY_ENERGY_FACTOR = 4


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


# ============================================================
# Training history
# ============================================================


@dataclass
class TrainingHistory:
    steps: np.ndarray
    energy: np.ndarray
    vscore: np.ndarray
    sigma: np.ndarray | None = None
    variance: np.ndarray | None = None
    error: np.ndarray | None = None  # |E - E_ref| / |E_ref|

    @classmethod
    def from_log(cls, log, N, E_ref=None, loss_name="Energy"):
        energy = np.array(log[loss_name]["Mean"]).real
        variance = np.real(np.array(log[loss_name]["Variance"]))
        return cls(
            steps=np.arange(len(energy)),
            energy=energy,
            vscore=N * variance / energy**2,
            sigma=np.array(log[loss_name]["Sigma"]).real,
            variance=variance,
            error=_rel_error(energy, E_ref),
        )


def _rel_error(energy, E_ref):
    if E_ref is None:
        return None
    return np.abs(np.asarray(energy) - E_ref) / np.abs(E_ref)


def plot_training(
    hist,
    *,
    title,
    E_global=None,
    E_irrep=None,
    irrep=None,
    info=None,
    schedule_setup=None,
    best_step=None,
    checkpoints=None,
    message=None,
    marker=None,
    xlim=None,
):
    """
    Energy / error / V-score of a training. The error panel is drawn when
    hist.error is available.

    best_step: index (in hist arrays) of the kept state, marked in gold
        (red when it had to be clamped to the last step).
    checkpoints: indices (in hist arrays) of the checkpointed steps.
    """
    has_error = hist.error is not None
    if has_error:
        fig, ax = plt.subplots(3, 1, figsize=(8, 18))
        ax_E, ax_err, ax_v = ax
    else:
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        (ax_E, ax_v), ax_err = ax, None

    steps = hist.steps
    n = len(steps)
    x_end = xlim if xlim is not None else n

    best_color = "gold"
    if best_step is not None and best_step >= n:
        best_step, best_color = n - 1, "red"

    def marks(axis, y):
        if best_step is not None:
            axis.plot(steps[best_step], y[best_step], marker="o", ms=4, color=best_color)
        if checkpoints is not None:
            axis.plot(
                steps[checkpoints], y[checkpoints], ls="", marker="o", ms=3, color="tan"
            )

    # Energy
    ax_E.set_title(title)
    if E_global is not None:
        if hist.sigma is not None:
            ax_E.errorbar(
                steps, hist.energy, yerr=hist.sigma, fmt="none", ecolor="r", label="E_stdev"
            )
        ax_E.hlines(E_global, 0, x_end, color="green", label="ED global")
        if E_irrep is not None:
            ax_E.hlines(E_irrep, 0, x_end, color="orange", ls="--", label=f"ED {irrep}")
    ax_E.plot(steps, hist.energy, color="blue", marker=marker, ms=3, label="E")
    marks(ax_E, hist.energy)
    if info:
        ax_E.text(
            0.9,
            0.9,
            info,
            transform=ax_E.transAxes,
            fontsize=10,
            color="k",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    if schedule_setup is not None:
        plot_schedule_setup(ax_E, schedule_setup)
    ax_E.set_ylabel("Energy", fontsize=12)

    # Error
    if has_error:
        ax_err.plot(steps, hist.error, color="red", label="E")
        marks(ax_err, hist.error)
        ax_err.set_yscale("log")
        ax_err.set_ylabel("Error", fontsize=12)

    # V-score
    ax_v.plot(steps, hist.vscore, color="purple", label="Vscore")
    marks(ax_v, hist.vscore)
    ax_v.set_yscale("log")
    ax_v.set_ylabel("Vscore", fontsize=12)

    for axis in ax:
        axis.legend()
        axis.set_xlabel("Iteration")
        axis.grid()
        if xlim is not None:
            axis.set_xlim(0, 1.01 * xlim)

    if message:
        ax[-1].text(0.6, -0.1, message, transform=ax[-1].transAxes)

    return fig, ax


def _info_text(E_best, time_exe, E_global=None, E_irrep=None, best_step=None):
    lines = []
    if E_global is not None:
        lines.append(f"E_gr global: {E_global:.4f}")
        if E_irrep is not None:
            lines.append(f"E_gr irrep: {E_irrep:.4f}")
    lines.append(f"E_best: {E_best:.4f}")
    lines.append(f"time_exe: {time_exe:.2f}")
    if best_step is not None:
        lines.append(f"Best step: {best_step:d}")
    return "\n".join(lines)


def save_history(hist, file_prefix):
    """Write the history as .txt files; returns {name: path}."""
    paths = {
        "energy": file_prefix + "_energy.txt",
        "energy_dev": file_prefix + "_energy_dev.txt",
        "vscore": file_prefix + "_vscore.txt",
    }
    np.savetxt(paths["energy"], hist.energy)
    np.savetxt(paths["energy_dev"], hist.sigma)
    np.savetxt(paths["vscore"], hist.vscore)
    if hist.error is not None:
        paths["error"] = file_prefix + "_error.txt"
        np.savetxt(paths["error"], hist.error)
    return paths


# ============================================================
# End-of-simulation callback plot
# ============================================================


def dump_callback(
    log,
    *,
    ed,
    labels,
    write_folder,
    size,
    time_exe,
    best_step,
    best_energy,
    schedule_setup=None,
    do_each_checkpoint=None,
    message="",
    write_callback=True,
):
    """
    Plot (and optionally save as .txt) the training history of the RuntimeLog.

    ed: EDResult; labels: SimLabels; best_step/best_energy: the state kept by
    BestIterKeeper (the one whose results are saved).
    Returns the artifact paths {"plot", "energy", "energy_dev", "vscore", "error"}.
    """
    os.makedirs(write_folder, exist_ok=True)

    N = int(np.prod(size))
    hist = TrainingHistory.from_log(log, N, E_ref=ed.E_ref)

    checkpoints = (
        np.arange(0, len(hist.energy), do_each_checkpoint)
        if do_each_checkpoint is not None
        else None
    )
    fig, _ = plot_training(
        hist,
        title=labels.title,
        E_global=ed.E_global,
        E_irrep=ed.E_irrep,
        irrep=ed.irrep,
        info=_info_text(best_energy, time_exe, ed.E_global, ed.E_irrep),
        schedule_setup=schedule_setup,
        best_step=best_step,
        checkpoints=checkpoints,
        message=message,
    )

    file_prefix = write_folder + "Callback_" + labels.sim
    artifacts = {"plot": file_prefix + ".jpeg"}
    fig.tight_layout()
    fig.savefig(artifacts["plot"], dpi=600)
    plt.close(fig)

    if write_callback:
        artifacts.update(save_history(hist, file_prefix))

    return artifacts


# ============================================================
# Plot from checkpoints (plot_checkpoint.py)
# ============================================================


def checkpoint_callback(artifact, steps, energy, vscore, artifact_path=None):
    """
    Same plot as dump_callback from checkpoint metrics (energy, vscore at the
    checkpointed steps). artifact: results JSON (or None).
    """
    hist_kwargs = dict(
        steps=np.asarray(steps), energy=np.asarray(energy), vscore=np.asarray(vscore)
    )

    if artifact is None:
        hist = TrainingHistory(**hist_kwargs)
        return plot_training(
            hist, title="Callback", marker="o", xlim=steps[-1], info=None
        )

    results = artifact["results"]
    factor = 1 if "energy_units" in results else LEGACY_ENERGY_FACTOR

    def scaled(key):
        v = results.get(key)
        return None if v is None else float(np.real(v)) * factor

    E_global, E_irrep = scaled("E_gr_global"), scaled("E_gr_irrep")
    E_ref = E_irrep if E_irrep is not None else E_global
    irrep = (results.get("irrep_info") or {}).get("irrep")

    hist = TrainingHistory(**hist_kwargs, error=_rel_error(energy, E_ref))

    return plot_training(
        hist,
        title=labels_from_artifact(artifact, artifact_path).title,
        E_global=E_global,
        E_irrep=E_irrep,
        irrep=irrep,
        info=_info_text(
            scaled("E_best"), results["time_exe"], E_global, E_irrep, results["best_step"]
        ),
        schedule_setup=results.get("schedule_flat"),
        marker="o",
        xlim=artifact["SIM"]["schedule"].get("total_epochs", steps[-1]),
    )
