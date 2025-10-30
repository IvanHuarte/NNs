import numpy as np
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import flax
import numpy.typing as npt
from typing import Optional
from pathlib import Path


class BestIterKeeper:
    """Store the values of a bunch of quantities from the best iteration.

    "Best" is defined in the sense of lowest energy.

    Args:
        Hamiltonian: An array containing the Hamiltonian matrix.
        N: The number of spins in the chain.
        baseline: A lower bound for the V score. If the V score of the best
            iteration falls under this threshold, the process will be stopped
            early.
        filename: Either None or a file to write the best state to.
    """

    def __init__(
        self,
        epochs: int,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float = 1e-8,
        filename: Optional[Path] = None,
        mode: str = "best_energy",
        balanced_setup: dict = {
            "start_stats": 2 / 10,
            "stats_window": 1 / 10,
            "normalizer": "Zscore",
            "selector": "linear",
        },
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf

        self.best_state_energy = np.inf
        self.best_state_vscore = np.inf
        self.best_step = 0
        self.best_state = None
        self.step_threshold = 0  # epochs // 10
        self.step = -1

        if mode == "best_energy":
            self.update = self.best_energy_update
        elif mode == "best_vscore":
            self.update = self.best_vscore_update
        elif mode == "always":
            self.update = self.always_update

    def best_energy_update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        self.step += 1

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = self.N * var / mean**2

        if self.step > self.step_threshold:
            if self.best_state_energy > energy_step:
                self.best_state = copy.copy(vstate)
                self.best_state_energy = energy_step
                self.best_state_vscore = vscore_step
                self.best_step = self.step

                if self.filename != None:
                    with open(self.filename, "wb") as file:
                        file.write(flax.serialization.to_bytes(driver.state))

        return self.survive_condition(energy_step, vscore_step)

    def best_vscore_update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        self.step += 1

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = self.N * var / mean**2

        if self.step > self.step_threshold:

            if self.best_state_vscore > vscore_step:
                self.best_state = copy.copy(driver.state)
                self.best_state_energy = energy_step
                self.best_state_vscore = vscore_step
                self.best_step = self.step

                if self.filename != None:
                    with open(self.filename, "wb") as file:
                        file.write(flax.serialization.to_bytes(driver.state))

        return self.survive_condition(energy_step, vscore_step)

    def always_update(self, step, log_data, driver):

        self.step += 1

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = self.N * var / mean**2

        # Always update

        if self.step > self.step_threshold:

            self.best_state = copy.copy(driver.state)
            self.best_state_energy = energy_step
            self.best_state_vscore = vscore_step
            self.best_step = self.step

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.survive_condition(energy_step, vscore_step)

    def balanced_update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = self.N * var / mean**2

        # print(f"\nStep {step}:")
        # print(f"Energy: {energy_step:.6e}, Vscore: {vscore_step:.6e}")
        if self.step > self.start_stats:

            if self.step > self.start_stats + self.stats_window:
                norm_E, norm_V = self.normalizer(energy_step, vscore_step)
                score = self.selector(norm_E, norm_V)
                # print(f"norm_E = {norm_E:.6e}, norm_V = {norm_V:.6e}")
                # print(f"Score: {score:.6f}, Best score: {self.best_score:.6f}")

                if score < self.best_score:
                    # print(f"New best score found: {score:.6f} < {self.best_score:.6f}. Updating best state.")
                    self.best_score = score
                    self.best_state = copy.copy(driver.state)
                    self.best_state_energy = energy_step
                    self.best_state_vscore = vscore_step
                    self.best_step = self.step

                    if self.filename != None:
                        with open(self.filename, "wb") as file:
                            file.write(flax.serialization.to_bytes(driver.state))

            self.normalizer.update_history(energy_step, vscore_step)
            # print(f"Updating historial. Length: {len(self.normalizer.energy_history)}")

        return self.survive_condition(energy_step, vscore_step)

    def survive_condition(self, energy, vscore):

        survive = True
        if vscore < self.baseline:
            survive = False
            self.exit_msg = f"Vscore {vscore} is below baseline {self.baseline}"
            print(self.exit_msg)
        if energy == float("nan"):
            survive = False
            self.exit_msg = f"Energy has diverged. Simulation crashed."
            print(self.exit_msg)

        return survive


class EnergyPlotter:
    def __init__(self, H, N, E_prev=None, E_ED=None, vs_prev=None, error_prev=None):
        self.H = H
        self.N = N
        self.energies = []
        self.vscores = []
        self.errors = []
        self.E_ED = E_ED

        # Configuro Matplotlib en modo interactivo
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=[18, 13])

        (self.energy,) = self.ax1.plot([], [], "-o", color="blue", label="Energía")
        (self.vscore,) = self.ax2.plot([], [], "-o", color="purple", label="Vscore")
        (self.error,) = self.ax3.plot([], [], "-o", color="red", label="Error")

        if E_ED is not None:
            self.ax1.axhline(
                E_ED, color="tab:green", linestyle="-", label="E_ED (Exact diag.)"
            )
        if E_prev is not None:
            self.ax1.axhline(
                E_prev,
                color="tab:orange",
                linestyle="--",
                alpha=0.8,
                label="Energía inicial",
            )
        if vs_prev is not None:
            self.ax2.axhline(
                vs_prev, color="tab:purple", linestyle="--", label="Vscore inicial"
            )
        if error_prev is not None:
            self.ax3.axhline(
                error_prev,
                color="tab:red",
                linestyle="--",
                alpha=0.8,
                label="Error inicial",
            )

        self.ax1.set_ylabel("Energía")
        self.ax1.set_title("Refinement callback")
        self.ax1.legend(fontsize=8)
        self.ax1.grid()

        self.ax2.set_ylabel("Vscore")
        self.ax2.set_yscale("log")
        self.ax2.legend(fontsize=8)
        self.ax2.grid()

        self.ax3.set_xlabel("Iteración")
        self.ax3.set_ylabel("Error")
        self.ax3.set_yscale("log")
        self.ax3.legend(fontsize=8)
        self.ax3.grid()

    def __call__(self, step, log_data, driver):
        # Calcula la energía y vstate en este step
        vstate = driver.state
        E = float(np.real(vstate.expect(self.H).mean))
        self.energies.append(E)
        # Vscore
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        self.vscores.append(self.N * var / mean**2)
        # Error
        self.errors.append(float(np.abs(E - self.E_ED) / np.abs(self.E_ED)))

        # Actualiza la curva
        self.energy.set_data(np.arange(len(self.energies)), self.energies)
        self.vscore.set_data(np.arange(len(self.vscores)), self.vscores)
        self.error.set_data(np.arange(len(self.errors)), self.errors)
        # self.ax1.text(1.05, 0.5, f"lr: {float(log_data["Optimizer/learning_rate"])}", transform=self.ax1.transAxes)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        # Dibuja y hace una pausa breve para que se renderice
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        return True


def dump_callback(logger, settings, write=False):

    callback_artifacts = {}

    time_exe = settings["time_exe"]
    write_folder = settings["write_folder"]
    architecture_display = settings["architecture"]
    opt_name = settings["opt_name"]
    learning_rate = settings["learning_rate"]
    sim_label = settings["sim_label"]
    title_label_callback = settings["title_label_callback"]
    best_step = settings["best_step"]

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

    setup_sim = f"E_best: {E_best:.4f} \nopt: {opt_name} \nl_rate: {learning_rate} \ntime_exe: {time_exe:.2f}"
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
    if architecture_display is not None:
        ax[0].text(
            0.45,
            0.85,
            architecture_display,
            transform=ax[0].transAxes,
            fontsize=12,
            color="k",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
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
