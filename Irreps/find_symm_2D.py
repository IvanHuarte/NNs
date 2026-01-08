#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

import json
from pathlib import Path
import sys

import jax
import jax.nn
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.typing
import netket as nk
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

cmap = colors.LinearSegmentedColormap.from_list(
    "Spectral_soft", sns.color_palette("Spectral", 256, desat=0.7)
)

matplotlib.rcParams["font.size"] = 12

from VA_project.initialize_model import ModelFactory
from VA_project.engine.runners import Runner
from Irreps.SymmBuilder import SymmGroup, _all_idx_combinations

path = str(Path(__file__).resolve().parent) + "/"
with open(path + "config.json", "r") as f:
    config = json.load(f)

write = config["write_folder"]
config_cm = config["config_cm"]
symmetries_config = config["symmetries"]

sizes = config_cm["sizes"]

for size in sizes:

    N = int(jnp.prod(jnp.array(size)))
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=None)

    model_factory = ModelFactory(size, config_cm)

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

        write_folder = write + f"{cm_model_name}/"
        folder_path = write + f"Size_{size[0]}x{size[1]}/"
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        print(f"\n************************ {cm_model_name} ************************")
        print(f"Size: {size}")
        print(f"Parameters: \n{params}")
        print(f"*************************************************************\n")

        cm_model = model_factory.get_model()
        eng = Runner(cm_model.cm, S_operators=model_factory.S_operators)

        # Build the Hamiltonian
        H = eng.build_hamiltonian(hilbert)

        configurations = hilbert.all_states()
        n_configurations = configurations.shape[0]

        group = SymmGroup(symmetries_config, lattice_size=size)

        # Generate the unitary representations for all symmetry operations
        representations = group.get_unitary_representations(configurations)

        # Compute irrep projectors
        q_idx_generator = _all_idx_combinations(group.N_group)

        Q = {}
        for q_vector in q_idx_generator:
            projector = group.get_irrep_projector(representations, q_vector)
            Q[q_vector] = sp.linalg.orth(projector)

        # Summarize the results.
        print("Dimensions of the projector basis:")
        for irrep in Q:
            print(f"{irrep}:\t{Q[irrep].shape[1]}")

        print("TOTAL:", sum(Q[irrep].shape[1] for irrep in Q))

        # Build the joint basis.
        adapted_basis = np.concatenate(list(Q.values()), axis=1)
        print("RANK OF THE BASIS MATRIX:", np.linalg.matrix_rank(adapted_basis))

        # Transform the Hamiltonian to the symmetry-adapted basis.
        H_dense = H.to_dense()
        adapted_matrix = adapted_basis.conj().T @ H_dense @ adapted_basis

        ############
        # PLOTTING
        ############

        # Visually check if the matrix is block-diagonal.
        abs_matrix = np.abs(adapted_matrix)

        # --- Figura 1: matriz adaptada ---
        fig1, ax1 = plt.subplots()
        cax = ax1.matshow(abs_matrix)
        fig1.colorbar(cax)
        count = 0
        for irrep in list(Q.keys())[:-1]:
            count += Q[irrep].shape[1]
            ax1.axhline(count - 0.5, color="white")
            ax1.axvline(count - 0.5, color="white")
        ax1.set_title("Symmetry-adapted Hamiltonian")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        # fig1.show()  # muestra la figura

        # --- Figura 2: eigenvalues ---
        fig2, ax2 = plt.subplots(figsize=(20, 20))

        eigvals = {}
        global_min = np.inf
        global_max = -np.inf
        for i, q_vector in enumerate(Q):
            basis = Q[q_vector]
            block = basis.conj().T @ H_dense @ basis
            eigvals[q_vector] = sp.linalg.eigvalsh(block)
            global_min = min(global_min, eigvals[q_vector].min())
            global_max = max(global_max, eigvals[q_vector].max())

        for i, q_vector in enumerate(Q):
            ax2.scatter(
                [i] * len(eigvals[q_vector]),
                eigvals[q_vector],
                c=eigvals[q_vector],
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
                label=f"{q_vector}",
                marker="o",
                s=100,
            )

        ax2.set_title(
            f"Eigenvalues per irrep {cm_model_name} {params} {size[0]}x{size[1]}"
        )
        ax2.set_xlabel(r"$q$", fontsize=15)
        ax2.set_ylabel(r"$E$", fontsize=15)
        ax2.set_xticks(range(len(Q)))
        ax2.set_xticklabels([str(q) for q in Q.keys()], rotation=45, ha="right")
        span = global_max - global_min
        ax2.hlines(global_min, -0.5, len(Q) - 0.5, linestyles="dashed", color="green")
        ax2.set_ylim(global_min - 0.2 * span, global_max + 0.2 * span)

        print("\nContinue [Enter] | Save&Continue [s] | Exit [q]: ")

        def on_key(event):
            if event.key == "q":
                print("Exit")
                plt.close("all")
                sys.exit(0)

            if event.key == "s":
                print("Save & continue")
                fig1.savefig(
                    folder_path + f"BlockHamiltonian_{sim_label}.png",
                    dpi=600,
                )
                fig2.savefig(
                    folder_path + f"SpectraPerIrrep_{sim_label}.png",
                    dpi=600,
                )
                plt.close("all")

            if event.key in "enter":
                print("Continue")
                plt.close("all")

        fig1.canvas.mpl_connect("key_press_event", on_key)
        fig2.canvas.mpl_connect("key_press_event", on_key)

        X1, Y1 = 1900, 200
        X2, Y2 = 0, 0

        manager = fig1.canvas.manager
        geom = manager.window.geometry()
        _, _, dx, dy = geom.getRect()
        print(dx, dy)
        manager.window.setGeometry(X1, Y1, int(1.5 * dx), int(1.5 * dy))

        manager = fig2.canvas.manager
        geom = manager.window.geometry()
        _, _, dx, dy = geom.getRect()
        manager.window.setGeometry(X2, Y2, dx, dy)

        plt.show()
