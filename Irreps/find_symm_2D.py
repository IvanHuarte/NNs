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

matplotlib.rcParams["font.size"] = 12

from VA_project.initialize_model import ModelFactory
from VA_project.engine.runners import Runner
from Irreps.SymmBuilder import SymmGroup, _all_idx_combinations

path = str(Path(__file__).resolve().parent ) + "/"
with open(path + "config.json", "r") as f:
    config = json.load(f)

write = config["write_folder"]
config_cm = config["config_cm"]
symmetries_config = config["symmetries"]

sizes = config_cm["sizes"]

for size in sizes:

    write_folder_size = write + f"Size_{size[0]}x{size[1]}/"

    N = int(jnp.prod(jnp.array(size)))
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=None)

    model_factory = ModelFactory(size, config_cm)

    for params in model_factory.get_params():

        cm_model_setup = model_factory.get_setup()
        cm_model_name = model_factory.name

        print(f"********************* {cm_model_name} *********************")
        print(f"Size: {size}")
        print(f"Parameters: \n{params}")
        print(f"*************************************************************")
        
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

        plt.matshow(abs_matrix)
        plt.colorbar()
        count = 0
        for irrep in list(Q.keys())[:-1]:
            count += Q[irrep].shape[1]
            plt.axhline(count - 0.5, color="white")
            plt.axvline(count - 0.5, color="white")
        plt.title("Symmetry-adapted Hamiltonian")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()


        # Extract the eigenvalues block by block.
        cmap = sns.color_palette("mako", as_cmap=True)
        valores = np.linspace(0, 1, len(Q))
        plt.figure(figsize=(20, 20))

        eigvals = {}

        global_min = np.inf
        global_max = -np.inf
        for i, q_vector in enumerate(Q):
            basis = Q[q_vector]
            block = basis.conj().T @ H_dense @ basis
            eigvals[q_vector] = sp.linalg.eigvalsh(block)
            global_min = min(global_min, eigvals[q_vector].min())
            global_max = max(global_max, eigvals[q_vector].max())

        norm = colors.Normalize(vmin=global_min, vmax=global_max)

        for i, q_vector in enumerate(Q):

            plt.scatter(
                [i]*len(eigvals[q_vector]),
                eigvals[q_vector],
                c = norm(eigvals[q_vector]),
                cmap=cmap,
                label=f"{q_vector}",
                marker='o',
                alpha=0.5,
                s=100,
            )
        plt.xlabel("q")
        plt.ylabel("E")
        plt.legend(loc="best")
        plt.xticks(range(len(Q)), [str(q) for q in Q.keys()], rotation=45, ha='right')

        span = global_max - global_min
        plt.ylim(global_min - 0.2 * span, global_max + 0.2 * span)
        plt.ylim(global_min - 0.2 * span, global_max + 0.2 * span)

        plt.tight_layout()
        plt.show()

        
        resp = input("Continue [c] | Save&Continue [sc] | Exit [e]: ").strip().lower()
        if resp in ("e", "s", "si", "yes"):
            print("Execution stopped")
            sys.exit(0)
        print("Saving....")


        


