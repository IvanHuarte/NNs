import ast
import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

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


def energyVSirrep(
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

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=[10, 8],
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    y_min = np.inf
    y_max = -np.inf

    E = []
    vscore = []
    irreps = []

    # print_tree(artifact_paths, values=True)
    # print(artifact_paths[str((0,0,0))])
    # print(static_args)
    # print(mode)
    print(static_args)

    for irrep, artifact_path in artifact_paths.items():

        assert (
            len(artifact_path) == 1
        ), f"Ambiguous files for irrep {irrep}: {artifact_path}"

        with open(artifact_path[0], "r") as f:
            artifact = json.load(f)

        # ENERGY
        E_gr = artifact["results"]["E_best"]
        # print(f"Energy for irrep {irrep}: {E_gr}")
        y_min = min(y_min, E_gr)
        y_max = max(y_max, E_gr)

        E.append(E_gr)
        vscore.append(artifact["results"]["vscore"])
        irreps.append(irrep)

    if plot_setup["split_by_parity"]:
        irreps = [ast.literal_eval(irrep) for irrep in irreps]
        E_even = [E[i] for i in range(len(E)) if irreps[i][2] == 0]
        E_odd = [E[i] for i in range(len(E)) if irreps[i][2] == 1]
        vscore_even = [vscore[i] for i in range(len(E)) if irreps[i][2] == 0]
        vscore_odd = [vscore[i] for i in range(len(E)) if irreps[i][2] == 1]
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

        print(f"Irreps: {irreps}")
        print(f"Irreps (even): {irreps_even}")
        print(f"Irreps (odd): {irreps_odd}")
        print(f"Set of irreps: {set_irreps}")
        print(f"Indices (even): {idx_even}")
        print(f"Indices (odd): {idx_odd}")

        ax2.set_xticks(
            range(len(set_irreps)), labels=set_irreps, rotation=45, fontsize=12
        )

        ax1.plot(
            range(len(idx_even)),
            E_even,
            color="red",
            marker="o",
            ms=4,
            alpha=0.5,
            label=r"$\text{Z_2 (+1)}$",
        )
        ax1.plot(
            range(len(idx_odd)),
            E_odd,
            color="blue",
            marker="o",
            ms=3,
            alpha=0.5,
            label=r"$\text{Z_2 (-1)}$",
        )
        ax2.plot(
            range(len(idx_even)),
            vscore_even,
            color="firebrick",
            marker="o",
            ms=4,
            alpha=0.5,
            label=r"$\text{Z_2 (+1)}$",
        )
        ax2.plot(
            range(len(idx_odd)),
            vscore_odd,
            color="purple",
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
            ax2.set_xticks(
                range(len(total_irreps)), labels=total_irreps, rotation=45, fontsize=12
            )
        else:
            ax2.set_xticks(range(len(irreps)), labels=irreps, rotation=45, fontsize=12)
        ax1.plot(irreps_idx, E, color=color, marker="o", ms=3, label=r"$Energy$")
        ax2.plot(
            irreps_idx, vscore, color="purple", marker="o", ms=3, label=r"$V-score$"
        )

    static_args_text = join_static_modes(static_args)
    ax1.text(
        0.1,
        0.94,
        rf"{static_args_text}",
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
        ),
        multialignment="center",
    )

    ax1.set_title(r"$Energy\;vs.\;Irrep$", fontsize=18)
    ax1.set_ylabel(r"$\langle H \rangle$", fontsize=18)
    ax2.set_xlabel(r"$Irrep$", fontsize=18)
    ax2.set_ylabel(r"$V-score$", fontsize=18)
    ax2.set_yscale("log")
    if x_lims is not None:
        ax1.set_xlim(*x_lims)
    if y_lims is not None:
        ax1.set_ylim(*y_lims)
    ax1.legend()
    ax1.grid()
    ax2.grid()
    plt.tight_layout()

    fig.savefig(
        write_folder + f"EnergyVSirrep{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
    print("\n\n")

    plt.close(fig)
