#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

import jax.numpy as jnp
import netket as nk
import numpy as np
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import json
from pathlib import Path

from NN_module.dynamic_plots import (
    batch_modphase_plotter,
    plot_M_sector_hist,
    plot_cum_contributors,
    plot_weighted_phase_hist,
)

matplotlib.rcParams["font.size"] = 12

cmap = colors.LinearSegmentedColormap.from_list(
    "Spectral_soft", sns.color_palette("Spectral", 256, desat=0.7)
)

from VA_project.initialize_model import ModelFactory
from VA_project.engine.runners import Runner

path = str(Path(__file__).resolve().parent) + "/"
with open(path + "config1_CM.json", "r") as f:
    config = json.load(f)

write = "/home/ihuarte/Escritorio/Ivan/NNs/Figures/"

sizes = config["sizes"]

for size in sizes:
    N = int(jnp.prod(jnp.array(size)))
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=None)

    model_factory = ModelFactory(size, config)

    for params in model_factory.get_params():

        cm_model_setup = model_factory.get_setup()
        cm_model_name = model_factory.name
        sim_label = "".join(
            f"_{key}_{value:.2f}"
            for key, value in params.items()
            if not isinstance(value, (tuple, list, dict))
        )
        title = (
            f"{cm_model_name} "
            + ", ".join(f"{key}={value}" for key, value in params.items())
            + f" Size: {size[0]}x{size[1]}"
        )

        write_folder = (
            write + f"{cm_model_name}/" + f"Size_{size[0]}x{size[1]}/" + sim_label + "/"
        )
        Path(write_folder).mkdir(parents=True, exist_ok=True)

        print(f"\n************************ {cm_model_name} ************************")
        print(f"Size: {size}")
        print(f"Parameters: \n{params}")
        print(f"*************************************************************\n")

        cm_model = model_factory.get_model()
        eng = Runner(cm_model.cm, S_operators=model_factory.S_operators)

        E_ED, x_ED = eng.exact_energy_lanczos(hilbert, k=10, eigenstates=True)

        sort_idx = np.argsort(E_ED)
        E_ED = E_ED[sort_idx]
        x_ED = x_ED.T[sort_idx]

        # Plot Spectrum
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(E_ED)), E_ED, color="blue", label="ED")
        ax.set_xlabel("State Index")
        ax.set_ylabel("Energy")
        ax.set_title(title)
        ax.grid()
        ax.legend()
        fig.savefig(write_folder + f"Spectrum{sim_label}.png", dpi=600)

        # Plot Modulus and Phase
        k = 3
        labels = [
            "GroundState",
            "First Excited",
            "Second Excited",
            "Third Excited",
            "Fourth Excited",
            "Fifth Excited",
            "Sixth_excited",
        ]
        Path(write_folder + "Modphase/").mkdir(parents=True, exist_ok=True)
        for i, (fig, ax) in enumerate(
            batch_modphase_plotter(x_ED[:k], labels=labels[:k], return_figure=True)
        ):
            fig.savefig(
                write_folder
                + "Modphase/"
                + f"{labels[i].replace(' ', '_')}{sim_label}.png",
                dpi=600,
            )

        # Ground State issues
        # Magnetization histogram
        fig_M, ax_M = plot_M_sector_hist(x_ED[0], hilbert)
        fig_M.suptitle(title, fontsize=30)

        fig_M.savefig(write_folder + f"Magnetization_histogram{sim_label}.png", dpi=600)

        # Cumulative contributors. Percentage of total components under modulus.
        # This is a way to see how many components are relevant in the state and
        # if there is a gap in the distribution of the modulus of the components.
        fig_cum, ax_cum = plot_cum_contributors(x_ED[0])
        fig_cum.savefig(
            write_folder + f"Cumulative_contributors{sim_label}.png", dpi=600
        )

        # Weighted phase histogram
        fig_phase, ax_phase = plot_weighted_phase_hist(x_ED[0])
        fig_phase.suptitle(title, fontsize=30)
        fig_phase.savefig(
            write_folder + f"Weighted_phase_histogram{sim_label}.png", dpi=600
        )

        plt.close("all")
