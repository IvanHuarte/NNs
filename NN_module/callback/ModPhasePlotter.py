import os
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from optax import scale

# matplotlib.rcParams["toolbar"] = "None"  # ← DESACTIVA icono SVG corrupto
# matplotlib.rcParams["figure.raise_window"] = False
# import warnings

# warnings.filterwarnings("ignore", message="qt.svg")

from NN_module.observables import full2red_basis_idx, modphase
from NN_module.label_utils import get_filenames_from_settings


class ModPhasePlotter:
    """
    Dynamic callback for plotting modulus and phase in each iteration
    """

    def __init__(
        self,
        sim_config,
        x_ED=None,
        no_null_mod=True,
        logscale=False,
        plot_each=10,
        savefig=False,
        sim_folder=None,
    ):
        self.plot_each = plot_each
        self.sim_config = sim_config
        self.x_ED = x_ED
        self.no_null_mod = no_null_mod
        self.logscale = logscale
        self.savefig = savefig
        self.write_folder = (
            sim_folder + "ModPhasePlotter/" if sim_folder is not None else None
        )
        (
            os.makedirs(self.write_folder, exist_ok=True)
            if self.write_folder is not None
            else None
        )
        size = sim_config["CM"]["size"]
        self.N = size[0] * size[1]

        if x_ED is not None:
            (mod_ED, phase_ED), stats_ED = modphase(x_ED)

        n_comp = x_ED.shape[0] if x_ED is not None else 4000
        msize = 0.1 * 65000 / n_comp
        self.margins = n_comp // 200

        _, _, _, callback = get_filenames_from_settings(
            sim_config["CM"], sim_config["NN"]
        )
        self.title = callback.replace("Callback", "").lstrip().replace(" ", "\\quad")

        # plt.ion()
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
            [], [], alpha=0.15, ls="", marker="o", ms=msize, color="r", label="vstate"
        )
        if x_ED is not None:
            self.sc_phase_ed = self.ax[2].plot(
                phase_ED, alpha=0.15, ls="", marker="o", ms=msize, label="ED"
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
        self.ax[0].set_title(
            r"$Modulus\;and\;Phase\qquad %s \qquad (step: 0)$" % (self.title),
            fontsize=10,
        )
        self.ax[0].set_xlim(-self.margins, n_comp + self.margins)
        self.ax[0].set_xticks([])
        self.ax[0].set_ylabel(r"$Modulus$")
        self.ax[0].legend(loc="upper right")

        self.ax[1].set_xlim(-self.margins, n_comp + self.margins)
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
            self.ax[2].set_xlim(-self.margins, n_comp + self.margins)

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

        self.max_mod_ED = max(mod_ED) if x_ED is not None else 0.0

        self.peak_texts_vs = []
        self.text_stats = None
        self.sample_vlines = []

        # Ajustes comunes
        for axis in self.ax:
            axis.grid(True)

        # plt.show()

    def __call__(self, step, log_data, driver):
        # Solo plotear cada plot_each iteraciones
        if step % self.plot_each != 0:
            return True

        self.ax[0].set_title(
            r"$Modulus\;and\;Phase\qquad %s \qquad (step: %d)$" % (self.title, step),
            fontsize=10,
        )

        vstate = driver.state
        (mod_vs, phase_vs), stats_vs = modphase(vstate)

        # Asegurarnos de usar numpy arrays para matplotlib (por si vienen jnp)
        mod_vs = np.asarray(mod_vs)
        phase_vs = np.asarray(phase_vs)

        # --- Actualizar línea de módulo del vstate ---
        x_mod = np.arange(len(mod_vs))
        self.line_mod_vs[0].set_data(x_mod, mod_vs)
        # Asegurar que el eje X muestre toda la señal
        self.ax[0].set_xlim(-self.margins, max(1, x_mod[-1]) + self.margins)
        # Actualizar límites Y a partir de los datos nuevos
        ymin_vlines_pos = -0.2 * max(self.max_mod_ED, float(np.max(mod_vs))) * 9 / 8
        self.ax[0].set_ylim(
            ymin_vlines_pos, max(self.max_mod_ED, float(np.max(mod_vs))) * 9 / 8
        )
        self.ax[0].relim()
        self.ax[0].autoscale_view(scalex=False, scaley=True)

        # --- Actualizar los samples (vlines) ---
        for vline in self.sample_vlines:
            try:
                vline.remove()
            except Exception:
                pass
        self.sample_vlines = []

        samples = vstate.samples.reshape(-1, self.N)
        # print(
        #     f"All sector 0: {jnp.allclose(
        #     jnp.sum(samples, axis=-1), jnp.zeros(samples.shape[0]), atol=1e-10
        # )}"
        # )
        bin_samples = (-(samples - 1) / 2).astype(jnp.int8)
        bin_samples = bin_samples.T[::-1].T
        powers = jnp.tile(jnp.arange(self.N), (bin_samples.shape[0], 1))
        samples_idx = jnp.sum(bin_samples * 2**powers, axis=-1)

        if vstate.hilbert._total_sz is not None:
            samples_idx = full2red_basis_idx(vstate.hilbert, samples_idx)

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

        # --- Actualizar scatter de fase del vstate (siguiendo tu enfoque con plot -> Line2D) ---
        # Si usas plot(..., ls="", marker="o") en init, self.sc_phase_vs es una lista con Line2D
        # y set_data funciona: aseguramos set_data y ajustamos xlim para que entren los puntos.
        x_vs = np.arange(len(phase_vs))
        self.sc_phase_vs[0].set_data(x_vs, phase_vs)
        self.ax[1].set_xlim(-self.margins, max(1, x_vs[-1]) + self.margins)
        # Mantener el eje Y fijado a [-pi, pi] (como en init), no autoscale en Y
        self.ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
        self.ax[1].relim()
        self.ax[1].autoscale_view(scalex=False, scaley=False)  # no cambiar Y

        # Si existiera scatter ED generado con plot en init, no lo tocamos (es estático)

        # --- Limpiar y redibujar el histograma en el último eje de forma robusta ---
        # Limpieza total del eje del histograma para evitar artefactos acumulados
        self.ax[-1].cla()

        # Re-configurar etiquetas/ticks del eje del histograma (igual que en __init__)
        self.ax[-1].set_xlabel(r"$Phase\;(radians)$")
        self.ax[-1].set_ylabel(r"$Phase \;histogram$")

        # reconstruir transform usado para escribir picos (necesario porque hicimos cla())
        self.transform = mtransforms.blended_transform_factory(
            self.ax[-1].transData, self.ax[-1].transAxes
        )

        if self.x_ED is not None:
            (mod_ED, phase_ED), stats_ED = modphase(self.x_ED)
            if self.no_null_mod:
                no_null_mod_mask = mod_ED > 1e-12
                phase_ED = phase_ED[no_null_mod_mask]
            # plot ED histogram (como en init)
            self.ax[-1].hist(
                np.asarray(phase_ED),
                bins=1000,
                range=(-np.pi, np.pi),
                density=True,
                alpha=0.7,
                label=f"ED  ({stats_ED['type']})",
            )
            # volver a dibujar picos ED si los hubiera
            if stats_ED.get("peaks") is not None:
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

        # Dibujar histograma del vstate
        self.phase_hist_vs = self.ax[-1].hist(
            np.asarray(phase_vs),
            bins=1000,
            range=(-np.pi, np.pi),
            color="r",
            density=True,
            alpha=0.7,
            label=f"vstate ({stats_vs['type']})",
        )

        # --- Texto con estadísticas ---
        if self.text_stats is not None:
            try:
                self.text_stats.remove()
            except Exception:
                pass

        self.text_stats = self.ax[-1].text(
            0.8,
            0.7,
            r"$\varphi_{vs}=%.2f \pm %.2f$"
            % (stats_vs["phase"]["mean"], stats_vs["phase"]["std"]),
            transform=self.ax[-1].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.4),
        )

        # --- Picos del vstate: borramos previos y dibujamos nuevos (sobre el eje ya limpiado) ---
        for txt in self.peak_texts_vs:
            try:
                txt.remove()
            except Exception:
                pass
        self.peak_texts_vs = []

        if stats_vs.get("peaks") is not None:
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

        # Leyenda del histograma
        try:
            self.ax[-1].legend(loc="upper right")
        except Exception:
            pass

        if self.logscale:
            self.ax[-1].set_yscale("log")

        # --- Redibujar canvas de forma eficiente ---
        # self.fig.canvas.draw_idle()
        # self.fig.canvas.flush_events()
        # plt.pause(0.01)

        # plt.ioff()
        if self.savefig:
            self.fig.savefig(
                self.write_folder + f"ModPhase_{step}.jpeg",
                dpi=600,
            )

        return True
