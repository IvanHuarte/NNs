import ast
import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from NN_module.saveNload import load_vstate

from ..utils import join_static_modes


def full_irreps(irrep_mode, size):

    symmetries = irrep_mode.split("_")[1:]

    for symm in symmetries:
        if symm == "Tx":
            Tx = range(size[0])
        elif symm == "Ty":
            Ty = range(size[1])
        elif symm == "Z2":
            Z2 = range(2)

    return list(product(Tx, Ty, Z2))


def fidelityVSirrep(
    artifact_paths,
    plot_setup,
    write_folder,
    static_args,
    mode,
    output_format="png",
):

    color = plot_setup["color"]
    x_lims = plot_setup["x_lims"]
    y_lims = plot_setup["y_lims"]

    sim_label = ""
    for args in static_args:
        sim_label += f"_{args[0]}_{args[1]}"
    sim_label += f"_runIn_{mode}"

    size = ast.literal_eval(static_args[0][1])
    if size[0] * size[1] > 12:
        print(
            f"WARNING: Skipping plot for size = {size}."
        )
        return

    fig, ax = plt.subplots(figsize=[10, 8])

    y_min = np.inf
    y_max = -np.inf

    fidelity = []
    irreps = []

    for irrep, artifact_path in artifact_paths.items():

        assert (
            len(artifact_path) == 1
        ), f"Ambiguous files for irrep {irrep}: {artifact_path}"

        with open(artifact_path[0], "r") as f:
            artifact = json.load(f)

        cm_model_name = artifact["CM"]["name"]
        size = artifact["CM"]["size"]
        params_str = ""
        for (param, value) in static_args:
            if param in ["size", "name"]:
                continue
            try:
                value = float(value) if "." in value else int(value)
                value = f"{value:.2f}" if isinstance(value, float) else str(value)
            except ValueError:
                pass
            
            params_str += f"_{param}_{value}"

        path_to_xED = f"/home/ihuarte/Escritorio/Ivan/NNs/Irreps/Figures/{cm_model_name}/Size_{size[0]}x{size[1]}/GroundState_{params_str}_irrep_{irrep}.txt"

        x_ED = np.loadtxt(path_to_xED, dtype=complex)
        vstate = load_vstate(artifact)
        fid = float(jnp.abs(jnp.vdot(vstate.to_array(), x_ED.squeeze())))

        y_min = min(y_min, fid)
        y_max = max(y_max, fid)

        fidelity.append(fid)
        irreps.append(irrep)

    if plot_setup["split_by_parity"]:
        irreps = [ast.literal_eval(irrep) for irrep in irreps]
        fidelity_even = [fidelity[i] for i in range(len(fidelity)) if irreps[i][2] == 0]
        fidelity_odd = [fidelity[i] for i in range(len(fidelity)) if irreps[i][2] == 1]
        irreps_even = [(irrep[0], irrep[1]) for irrep in irreps if irrep[2] == 0]
        irreps_odd = [(irrep[0], irrep[1]) for irrep in irreps if irrep[2] == 1]

        irreps_even = (
            full_irreps(mode.split("Z2")[0], artifact["CM"]["size"])
            if plot_setup["show_all_irreps"]
            else irreps_even
        )
        irreps_odd = (
            full_irreps(mode.split("Z2")[0], artifact["CM"]["size"])
            if plot_setup["show_all_irreps"]
            else irreps_odd
        )
        set_irreps = sorted(set(irreps_even + irreps_odd))
        idx_even = [set_irreps.index(irrep) for irrep in irreps_even]
        idx_odd = [set_irreps.index(irrep) for irrep in irreps_odd]

        # print(f"Irreps: {irreps}")
        # print(f"Irreps (even): {irreps_even}")
        # print(f"Irreps (odd): {irreps_odd}")
        # print(f"Set of irreps: {set_irreps}")
        # print(f"Indices (even): {idx_even}")
        # print(f"Indices (odd): {idx_odd}")

        ax.set_xticks(
            range(len(set_irreps)), labels=set_irreps, rotation=45, fontsize=12
        )

        ax.plot(
            range(len(idx_even)),
            fidelity_even,
            color="red",
            marker="o",
            ms=4,
            alpha=0.5,
            label=r"$\text{Z_2 (+1)}$",
        )
        ax.plot(
            range(len(idx_odd)),
            fidelity_odd,
            color="blue",
            marker="o",
            ms=3,
            alpha=0.5,
            label=r"$\text{Z_2 (-1)}$",
        )

    else:
        total_irreps = full_irreps(mode, artifact["CM"]["size"])
        total_irreps = [str(irrep) for irrep in total_irreps]
        # print(f"Total irreps: {total_irreps}")
        irreps_idx = (
            [total_irreps.index(irrep) for irrep in irreps]
            if plot_setup["show_all_irreps"]
            else range(len(irreps))
        )
        if plot_setup["show_all_irreps"]:
            ax.set_xticks(
                range(len(total_irreps)), labels=total_irreps, rotation=45, fontsize=12
            )
        else:
            ax.set_xticks(range(len(irreps)), labels=irreps, rotation=45, fontsize=12)
        ax.plot(irreps_idx, fidelity, color=color, marker="o", ms=3, label=r"$Fidelity$")
        

    static_args_text = join_static_modes(static_args)
    ax.text(
        0.1,
        0.94,
        rf"{static_args_text}",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
        ),
        multialignment="center",
    )

    ax.set_title(r"$Fidelity\;vs.\;Irrep$", fontsize=18)
    ax.set_ylabel(r"$\langle \psi_{VS} | \psi_{GS} \rangle$", fontsize=18)
    ax.set_xlabel(r"$Irrep$", fontsize=18)
    if x_lims is not None:
        ax.set_xlim(*x_lims)
    if y_lims is not None:
        ax.set_ylim(*y_lims)
    ax.legend()
    ax.grid()
    plt.tight_layout()

    fig.savefig(
        write_folder + f"FidelityVSirrep{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)
