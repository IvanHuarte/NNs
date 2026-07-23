import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["toolbar"] = "None"  # ← DESACTIVA icono SVG corrupto
matplotlib.rcParams["figure.raise_window"] = False
import warnings

warnings.filterwarnings("ignore", message="qt.svg")


class EnergyPlotter:
    def __init__(
        self,
        H,
        N,
        E_prev=None,
        E_ED=None,
        vs_prev=None,
        error_prev=None,
        savefig=False,
        sim_folder=None,
    ):
        self.H = H
        self.N = N
        self.energies = []
        self.vscores = []
        self.errors = []
        self.E_ED = E_ED
        self.savefig = savefig
        self.write_folder = (
            sim_folder + "EnergyPlotter/" if sim_folder is not None else None
        )
        (
            os.makedirs(self.write_folder, exist_ok=True)
            if self.write_folder is not None
            else None
        )

        # Configuro Matplotlib en modo interactivo
        # plt.ion()
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

        self.ax1.set_title(f"Simulation Callback - step {step}")

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
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # plt.pause(0.01)

        if self.savefig:
            self.fig.savefig(
                self.write_folder + f"Callback_{step}.jpeg",
                dpi=600,
            )
        return True
