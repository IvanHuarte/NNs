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

        # Compute irrep projectors
        q_idx_generator = (
            _all_idx_combinations(group.N_group)
            if q_idx_generator is None
            else q_idx_generator
        )
        Q = {}
        eigvals = {}
        for i, q_vector in enumerate(q_idx_generator):
            print(f"Processing q_vector: {q_vector}")
            projector = group.get_irrep_projector(representations, q_vector)

            basis = sp.linalg.orth(projector)
            block = basis.conj().T @ H_dense @ basis
            evals, evects = sp.linalg.eigh(
                block,
            )
            x_ED = basis @ evects[:, 0]
            # print(f"Projector SHAPE: {projector.shape}")
            # print(f"basis SHAPE: {basis.shape}")
            # print(f"EVECS SHAPE: {evects.shape}")


            # E_D = x_ED.conj().T @ H_dense @ x_ED
            # print(f"E_gr_irrep: {evals[0]}  |  E_gr: {E_D}")

            Q[q_vector] = basis
            eigvals[q_vector] = evals
            np.savetxt(
                folder_path + f"Spectrum_{sim_label}_irrep_{q_vector}.txt",
                evals,
            )
            np.savetxt(
                folder_path + f"GroundState_{sim_label}_irrep_{q_vector}.txt",
                x_ED,
            )
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

        # --- Figura 2: eigenvalues ---
        fig2, ax2 = plt.subplots(figsize=(20, 20))

        global_min = np.inf
        global_max = -np.inf
        for i, q_vector in enumerate(Q):

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
        ax2.hlines(global_min, - 0.5, len(Q) - 0.5, linestyles="dashed", color="green")
        ax2.set_ylim(global_min - 0.2 * span, global_max + 0.2 * span)

        group_label = "".join([str(g) for g in group.group_label])

        # Save manually
        fig2.savefig(
            folder_path + f"SpectraPerIrrep_{group_label}_{sim_label}.png",
            dpi=600,
        )
