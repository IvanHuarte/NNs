#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import json
from pathlib import Path
import sys

import jax.numpy as jnp
import jax.numpy.linalg as jla
import netket as nk
import numpy as np
import scipy as sp
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
from Irreps.utils import estimate_dim_q, build_irrep_basis

path = str(Path(__file__).resolve().parent) + "/"
with open(path + "config.json", "r") as f:
    config = json.load(f)

write = config["write_folder"]
config_cm = config["config_cm"]
symmetries_config = config["symmetries"]

sizes = config_cm["sizes"]

k_max = config["k_max"]

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


        # --------------------------------------------------
        # Construcción del Hamiltoniano y del grupo
        # --------------------------------------------------

        H = eng.build_hamiltonian(hilbert)
        H_dense = H.to_dense() if hasattr(H, "to_dense") else H

        configurations = hilbert.all_states()
        D = configurations.shape[0]

        group = SymmGroup(
            symmetries_config,
            lattice_size=size,
            all_states=configurations
        )

        q_idx_generator = _all_idx_combinations(group.N_group)

        # --------------------------------------------------
        # Diagonalización por irreps
        # --------------------------------------------------

        eigvals = {}
        irrep_dim = {}

        global_min = np.inf
        global_max = -np.inf
        for q_vector in q_idx_generator:
            print(f"\nirrep: {q_vector}")

            projector = group.get_irrep_projector_action(q_vector, norm=True)

            v = np.random.randn(D) + 1j*np.random.randn(D)
            lhs = projector(H @ v)
            rhs = H @ projector(v)
            print("|| [H, P] v || =", np.linalg.norm(lhs - rhs))

            # -------- construir base de la irrep --------
            print(f"Building irrep basis...")
            Qq = build_irrep_basis(projector, D)
            dim_q = Qq.shape[1]
            irrep_dim[q_vector] = dim_q 
            print("dim(q) =", dim_q)

            # -------- sanity checks --------
            err_proj = np.linalg.norm(projector(Qq[:, 0]) - Qq[:, 0])
            err_orth = np.linalg.norm(Qq.conj().T @ Qq - np.eye(dim_q))

            print("||Pψ - ψ|| =", err_proj)
            print("||Q†Q - I|| =", err_orth)

            # -------- Hamiltoniano restringido --------

            Hq = Qq.conj().T @ H_dense @ Qq

            # -------- diagonalización --------
            print("Diagonalizing...")
            eigvals[q_vector] = np.linalg.eigvalsh(Hq)[:k_max]
            print("E =", eigvals[q_vector])
            np.savetxt(folder_path + f"Spectrum_{sim_label}_irrep_{q_vector}.txt", eigvals[q_vector])

            global_min = min(global_min, eigvals[q_vector].min())
            global_max = max(global_max, eigvals[q_vector].max())


        # --- Figura 2: eigenvalues ---
        fig2, ax2 = plt.subplots(figsize=(20, 20))

        for i, q_vector in enumerate(eigvals.keys()):
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
        ax2.set_xticks(range(len(eigvals)))
        ax2.set_xticklabels([str(q) for q in eigvals.keys()], rotation=45, ha="right")
        span = global_max - global_min
        ax2.hlines(global_min, -0.5, len(eigvals) - 0.5, linestyles="dashed", color="green")
        ax2.set_ylim(global_min - 0.2 * span, global_max + 0.2 * span)

        # Save manually
        # fig1.savefig(
        #     folder_path + f"BlockHamiltonian_{sim_label}.png",
        #     dpi=600,
        # )
        fig2.savefig(
            folder_path + f"SpectraPerIrrep_{sim_label}.png",
            dpi=600,
        )
        exit(0)

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
