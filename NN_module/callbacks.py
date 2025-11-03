import numpy as np
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import flax
import jax.numpy as jnp
import numpy.typing as npt
from typing import Optional
from pathlib import Path

from NN_module.NN_utils import modphase
from NN_module.label_utils import get_filenames_from_settings


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
        self.step_threshold = epochs // 10
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


class ModPhasePlotter:
    """
    Dynamic callback for plotting modulus and phase in each iteration
    """

    def __init__(self, sim_config, x_ED=None, plot_each=5):
        self.plot_each = plot_each
        self.sim_config = sim_config
        self.x_ED = x_ED
        size = sim_config["CM"]["size"]
        self.N = size[0] * size[1]

        (mod_ED, phase_ED), stats_ED = modphase(x_ED)

        kwargs = {**sim_config["CM"], **sim_config["NN"]["setup"]}
        _, _, _, callback = get_filenames_from_settings(
            sim_config["CM"]["name"], sim_config["NN"]["name"], **kwargs
        )
        title = callback.replace("Callback", "").lstrip().replace(" ", "\\quad")

        plt.ion()
        nplots = 4 if x_ED is not None else 3
        self.fig, self.ax = plt.subplots(nplots, 1, figsize=[15, 10])

        # Plot MODULUS
        self.line_mod_vs = self.ax[0].plot(
            [], [], ls="-", color="r", alpha=0.6, label="vstate"
        )

        if x_ED is not None:
            self.line_mod_ed = self.ax[0].plot(mod_ED, ls="-", alpha=0.6, label="ED")
        else:
            self.line_mod_ed = None

        # Plot PHASE SCATTER
        self.sc_phase_vs = self.ax[1].plot(
            [], [], alpha=0.15, ls="", marker="o", ms=0.1, color="r", label="vstate"
        )
        if x_ED is not None:
            self.sc_phase_ed = self.ax[2].plot(
                phase_ED, alpha=0.15, ls="", marker="o", ms=0.1, label="ED"
            )
        else:
            self.sc_phase_ed = None

        # Plot PHASE HISTOGRAM
        self.phase_hist_vs = None

        if x_ED is not None:
            self.phase_hist_ed = self.ax[-1].hist(
                phase_ED,
                bins=1000,
                range=(-np.pi, np.pi),
                density=True,
                alpha=0.7,
                label=f"ED  ({stats_ED['type']})",
            )
        else:
            self.sc_phase_ed = None

        # Configuración de ejes
        self.ax[0].set_title(r"$Modulus\;and\;Phase\qquad %s$" % (title), fontsize=10)
        self.ax[0].set_xticks([])
        self.ax[0].set_ylabel(r"$Modulus$")
        self.ax[0].legend(loc="upper right")

        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax[1].set_yticklabels(
            [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
        )
        self.ax[1].set_ylabel(r"$Phase \;vstate$")
        self.ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
        self.ax[1].legend(loc="upper right")

        if x_ED is not None:
            self.ax[2].set_xlabel(r"$C_i$")
            self.ax[2].set_ylabel(r"$Phase \;ED$")
            self.ax[2].set_ylim(-np.pi - 0.1, np.pi + 0.1)
            self.ax[2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            self.ax[2].set_yticklabels(
                [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
            )
            self.ax[2].legend(loc="upper right")
        else:
            self.ax[1].set_xlabel(r"$C_i$")

        self.ax[-1].set_xlabel(r"$Phase\;(radians)$")
        self.ax[-1].set_ylabel(r"$Phase \;histogram$")

        self.transform = mtransforms.blended_transform_factory(
            self.ax[-1].transData, self.ax[-1].transAxes
        )

        if x_ED is not None and stats_ED["peaks"] is not None:
            for peak in stats_ED["peaks"]["values"]:
                self.ax[-1].text(
                    peak - 0.1,
                    0.9,
                    r"%.2f" % peak,
                    color="b",
                    transform=self.transform,
                    fontsize=8,
                    alpha=0.7,
                )

        self.max_mod_ED = max(mod_ED)
        self.peak_texts_vs = []
        self.text_stats = None
        self.sample_vlines = []

        # Ajustes comunes
        for axis in self.ax:
            axis.grid(True)

        plt.show()

    def __call__(self, step, log_data, driver):
        # Solo plotear cada plot_each iteraciones
        if step % self.plot_each != 0:
            return True

        vstate = driver.state
        (mod_vs, phase_vs), stats_vs = modphase(vstate)

        # Actualizar línea de módulo del vstate
        self.line_mod_vs[0].set_data(np.arange(len(mod_vs)), mod_vs)

        # Actualizar los samples
        # Borrar antiguas
        for vline in self.sample_vlines:
            vline.remove()
        self.sample_vlines = []

        samples = vstate.samples.reshape(-1, self.N)
        bin_samples = (-(samples - 1) / 2).astype(jnp.int8)
        bin_samples = bin_samples.T[::-1].T
        powers = jnp.tile(jnp.arange(self.N), (bin_samples.shape[0], 1))
        samples_idx = jnp.sum(bin_samples * 2**powers, axis=-1)

        # Agrega líneas verticales en las posiciones de samples_idx
        ymin_vlines_pos = -0.2 * max(self.max_mod_ED, max(mod_vs)) * 9 / 8
        for idx in np.asarray(samples_idx):
            vline = self.ax[0].vlines(
                x=idx,
                ymin=ymin_vlines_pos,
                ymax=0,
                color="gray",
                alpha=0.18,
                linewidth=0.5,
            )
            self.sample_vlines.append(vline)

        # Actualizar scatter de fase del vstate
        self.sc_phase_vs[0].set_data(np.arange(len(phase_vs)), phase_vs)

        # Actualizar límites de ejes
        self.ax[0].set_ylim(ymin_vlines_pos, max(self.max_mod_ED, max(mod_vs)) * 9 / 8)
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        self.ax[1].relim()
        self.ax[1].autoscale_view()

        # Limpiar histogramas anteriores
        if self.phase_hist_vs is not None:
            for patch in self.phase_hist_vs[2]:
                patch.remove()

        self.phase_hist_vs = self.ax[-1].hist(
            phase_vs,
            bins=1000,
            range=(-np.pi, np.pi),
            color="r",
            density=True,
            alpha=0.7,
            label=f"vstate ({stats_vs['type']})",
        )

        # Actualizar texto con estadísticas
        if self.text_stats is not None:
            self.text_stats.remove()

        self.text_stats = self.ax[-1].text(
            0.8,
            0.7,
            r"$\varphi_{vs}=%.2f \pm %.2f$"
            % (stats_vs["phase"]["mean"], stats_vs["phase"]["std"]),
            transform=self.ax[-1].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.4),
        )

        # Actualizar picos del vstate
        for txt in self.peak_texts_vs:
            txt.remove()
        self.peak_texts_vs = []

        if stats_vs["peaks"] is not None:
            for peak in stats_vs["peaks"]["values"]:
                txt = self.ax[-1].text(
                    peak - 0.1,
                    0.85,
                    r"%.2f" % peak,
                    color="r",
                    transform=self.transform,
                    fontsize=8,
                    alpha=0.7,
                )
                self.peak_texts_vs.append(txt)
        self.ax[-1].legend(loc="upper right")

        # Una sola actualización del canvas
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
