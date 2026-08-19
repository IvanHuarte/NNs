#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.nn
import jax.numpy as jnp
import jax.typing
import matplotlib
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.linalg
import seaborn as sns
from matplotlib import colors

cmap = colors.LinearSegmentedColormap.from_list(
    "Spectral_soft", sns.color_palette("Spectral", 256, desat=0.7)
)

matplotlib.rcParams["font.size"] = 12

from VA_project.engine.runners import Runner
from VA_project.initialize_model import ModelFactory

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

        q_idx_generator = config["calc_irreps"]
        if q_idx_generator is not None:
            q_idx_generator = [tuple(irrep) for irrep in q_idx_generator]

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
        folder_path = write_folder + f"Size_{size[0]}x{size[1]}/"
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        print(f"\n************************ {cm_model_name} ************************")
        print(f"Size: {size}")
        print(f"Parameters: \n{params}")
        print(f"Irreps: \n{q_idx_generator}")
        print("*************************************************************\n")

        cm_model = model_factory.get_model()
        eng = Runner(cm_model.cm, S_operators=model_factory.S_operators)
        E_ED = eng.exact_energy_lanczos(hilbert, k=5)
        print(f"Full H: {np.sort(E_ED)}")

        # Build the Hamiltonian
        H = eng.build_hamiltonian(hilbert)
        H_dense = H.to_dense()

        configurations = hilbert.all_states()
        n_configurations = configurations.shape[0]

        print("Initializing symmetry group...")

        group = SymmGroup(
            symmetries_config, lattice_size=size, all_states=configurations
        )

        print("Calculation unitary representations...\n")
        # Generate the unitary representations for all symmetry operations
        representations = group.get_unitary_representations(configurations)

        n_symmetries = len(representations)
        all_conmutes = True
        for i in range(n_symmetries):
            U = representations[i]
            conmutes = np.allclose(
                H_dense @ U - U @ H_dense, np.zeros(H_dense.shape, dtype=np.complex128)
            )
            print(f"Conmutes with H?: {conmutes}")
            all_conmutes = all_conmutes and conmutes

        print(f"All symmetries commute with H?: {all_conmutes}")
        print(
            np.unravel_index(
                np.arange(n_symmetries * n_symmetries), (n_symmetries, n_symmetries)
            )
        )

        if n_symmetries > 1:
            for i, j in zip(
                *np.unravel_index(
                    np.arange(n_symmetries * n_symmetries), (n_symmetries, n_symmetries)
                )
            ):
                U_i, U_j = representations[i], representations[j]
                conmutes = np.allclose(
                    U_i @ U_j - U_j @ U_i, np.zeros(U_i.shape, dtype=np.complex128)
                )
                print(f"Conmutes U_{i} with U_{j}?: {conmutes}")

        # Compute irrep projectors
        q_idx_generator = (
            _all_idx_combinations(group.N_group)
            if q_idx_generator is None
            else q_idx_generator
        )
        Q = {}
        for i, q_vector in enumerate(q_idx_generator):
            print(f"Processing q_vector: {q_vector}")
            projector = group.get_irrep_projector(representations, q_vector)
            # print(
            #     f"Conmutes?: {np.allclose(H_dense @ projector - projector @ H_dense, np.zeros(H_dense.shape, dtype=complex))}"
            # )
            # print(np.linalg.matrix_rank(projector))

            # np.savetxt(folder_path + f"{sim_label}_irrep_{q_vector}_projector.txt", projector)

            # print("Estimating irrep dimension...")
            # irrep_dim_est = int(
            #     round(np.real(np.trace(projector)) / group._group_norm())
            # )
            # print(f"Building irrep basis via svds_orth k={irrep_dim_est}\n")
            # Q[q_vector] = svds_orth(projector, k=irrep_dim_est)
            Q[q_vector] = sp.linalg.orth(projector)
            # np.savetxt(folder_path + f"{sim_label}_irrep_{q_vector}_Qmatrix.txt", Q[q_vector])

        # Summarize the results.
        print("Dimensions of the projector basis:")
        for irrep in Q:
            print(f"{irrep}:\t{Q[irrep].shape[1]}")

        print("TOTAL:", sum(Q[irrep].shape[1] for irrep in Q))

        # Build the joint basis.
        adapted_basis = np.concatenate(list(Q.values()), axis=1)
        print("RANK OF THE BASIS MATRIX:", np.linalg.matrix_rank(adapted_basis))

        # Transform the Hamiltonian to the symmetry-adapted basis.
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

        if all_conmutes:
            conmute_label = r"$[H,G]\;=\;0$"
        else:
            conmute_label = r"$[H,G]\;\neq\;0$"

        ax1.text(
            0.5,
            -0.1,
            conmute_label,
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=10,
        )

        # --- Figura 2: eigenvalues ---
        fig2, ax2 = plt.subplots(figsize=(20, 20))

        eigvals = {}
        global_min = np.inf
        global_max = -np.inf
        for i, q_vector in enumerate(Q):
            print(f"Diagonalizing {q_vector}...")
            basis = Q[q_vector]
            block = basis.conj().T @ H_dense @ basis
            evals, evect = sp.linalg.eigh(
                block,
            )
            eigvals[q_vector] = evals
            # print(evals.shape)
            # print(evect.shape)
            np.savetxt(
                folder_path + f"Spectrum_{sim_label}_irrep_{q_vector}.txt",
                eigvals[q_vector],
            )

            # print("E =", eigvals[q_vector][:5])
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
            irrep_gr = min(eigvals[q_vector])
            irrep_max = max(eigvals[q_vector])
            interval = irrep_max - irrep_gr
            ax2.text(i - 0.3, irrep_gr - interval / 20, f"{irrep_gr:.4f}")

        ax2.set_title(
            f"Eigenvalues per irrep {cm_model_name} {params} {size[0]}x{size[1]}"
        )
        ax2.set_xlabel(r"$q$", fontsize=15)
        ax2.set_ylabel(r"$E$", fontsize=15)
        ax2.set_xticks(range(len(Q)))
        ax2.set_xticklabels([str(q) for q in Q], rotation=45, ha="right")
        span = global_max - global_min
        ax2.hlines(global_min, -0.5, len(Q) - 0.5, linestyles="dashed", color="green")
        ax2.set_ylim(global_min - 0.2 * span, global_max + 0.2 * span)

        group_label = "".join([str(g) for g in group.group_label])

        # Save manually
        fig1.savefig(
            folder_path + f"BlockHamiltonian_{group_label}_{sim_label}.png",
            dpi=600,
        )
        fig2.savefig(
            folder_path + f"SpectraPerIrrep_{group_label}_{sim_label}.png",
            dpi=600,
        )
        continue

        print("\nContinue [Enter] | Save&Continue [s] | Exit [q]: ")

        def on_key(event):
            if event.key == "q":
                print("Exit")
                plt.close("all")
                sys.exit(0)

            if event.key == "s":
                print("Save & continue")
                fig1.savefig(
                    folder_path + f"BlockHamiltonian_{group_label}_{sim_label}.png",
                    dpi=600,
                )
                fig2.savefig(
                    folder_path + f"SpectraPerIrrep_{group_label}_{sim_label}.png",
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
