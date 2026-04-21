#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

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
import json
from pathlib import Path


cmap = colors.LinearSegmentedColormap.from_list(
    "Spectral_soft", sns.color_palette("Spectral", 256, desat=0.7)
)

matplotlib.rcParams["font.size"] = 12

from Irreps.utils import svds_orth
from VA_project.initialize_model import ModelFactory
from VA_project.engine.runners import Runner
from Irreps.SymmBuilder import SymmGroup, _all_idx_combinations

path = str(Path(__file__).resolve().parent) + "/Irreps/"
with open(path + "config.json", "r") as f:
    config = json.load(f)

write = "/home/ihuarte/Escritorio/Ivan/NNs/Irreps/Figures/"
config_cm = config["config_cm"]
symmetries_config = config["symmetries"]

sizes = config_cm["sizes"]

size = sizes[0]
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
    H_dense = H.to_dense()

    configurations = hilbert.all_states()   
    n_configurations = configurations.shape[0]

    symmetries_config["Traslation"]["on"] = False
    symmetries_config["Cn"]["on"] = True
    group = SymmGroup(
        symmetries_config, 
        lattice_size=size,
        all_states=configurations
    )
    C2 = group.get_unitary_representations(configurations)

    eigvals = {}    
    global_min = np.inf
    global_max = -np.inf
    ### Load Q
    print("Loading Q files")
    irreps = [(0,0), (2,2), (0,2)]
    Q = {}
    for i, q_vector in enumerate(irreps):
        print({q_vector})
        
        Q_k = np.loadtxt(folder_path + f"{sim_label}_irrep_{q_vector}_Qmatrix.txt", dtype=complex)
        if q_vector in [(0,0), (size[0]//2,size[1]//2)]:
            C2_k = (Q_k.T.conj() @ C2 @ Q_k).squeeze()
        else:
            Q_mk = np.loadtxt(folder_path + f"{sim_label}_irrep_{q_vector[::-1]}_Qmatrix.txt", dtype=complex)
            Off_D = (Q_k.T.conj() @ C2 @ Q_mk).squeeze()
            C2_k = np.block([
                [np.zeros_like(Off_D), Off_D]
                [Off_D.T.conj(), np.zeros_like(Off_D)]
            ])
        
        print(f"Diagonalizing C2_k with shape: {C2_k.shape}")
        C2_evals, C2_evec = scipy.linalg.eigh(C2_k)

        plus_mask = np.isclose(C2_evals, 1.0)
        minus_mask = np.isclose(C2_evals, -1.0)
        V_C2_plus = C2_evec[:, plus_mask]
        V_C2_minus = C2_evec[:, minus_mask]

        for ir, V in enumerate([V_C2_plus, V_C2_minus]):
            q_vect_C2 = q_vector + (ir,)
            print(f"Diagonalizing {q_vect_C2}...")
            basis = Q_k @ V
            block = basis.conj().T @ H_dense @ basis
            eigvals[q_vect_C2] = sp.linalg.eigvalsh(block)
            np.savetxt(folder_path + f"TC2_Spectrum_{sim_label}_irrep_{q_vect_C2}.txt", eigvals[q_vect_C2])

            print("E =", eigvals[q_vect_C2])
            global_min = min(global_min, eigvals[q_vect_C2].min())
            global_max = max(global_max, eigvals[q_vect_C2].max())


    # --- Figura 2: eigenvalues ---
    fig2, ax2 = plt.subplots(figsize=(20, 20))
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
        f"Eigenvalues per irrep T-C2 {cm_model_name} {params} {size[0]}x{size[1]}"
    )
    ax2.set_xlabel(r"$q$", fontsize=15)
    ax2.set_ylabel(r"$E$", fontsize=15)
    ax2.set_xticks(range(len(Q)))
    ax2.set_xticklabels([str(q) for q in Q.keys()], rotation=45, ha="right")
    span = global_max - global_min
    ax2.hlines(global_min, -0.5, len(Q) - 0.5, linestyles="dashed", color="green")
    # ax2.set_ylim(global_min - 0.2 * span, global_max + 0.2 * span)
    ax2.set_ylim(global_min - 0.1 * span, -4.2)


    # Save manually

    fig2.savefig(
        folder_path + f"SpectraPerIrrep_{sim_label}.png",
        dpi=600,
    )