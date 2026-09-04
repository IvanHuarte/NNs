import numpy as np
import scipy as sp

from Irreps.SymmBuilder import SymmGroup

equivariant_arch = ["Sequivariant", "SequivariantX"]

symm_config = {"Traslation": {"on": False, "subgroup": [0, 1]}, "Z2": {"on": False}}


def recursive_value_from_key(tree, key):

    for k, v in tree.items():

        if k == key:
            return v
        elif isinstance(v, dict):
            result = recursive_value_from_key(v, key)
            if result is not None:
                return result

    return None


def check_irreps(config_nn):
    symmetries = []
    irrep = []

    experiment_selection = config_nn["selection"]
    symm_wrapper = config_nn["symm_wrapper"]
    main_arch = config_nn[experiment_selection]["stage0"]["template"]

    # Traslation symmetrization
    if symm_wrapper["Traslation"]["on"]:
        irrep.extend(symm_wrapper["Traslation"]["irrep"])
        symmetries.extend(["Tx", "Ty"])

    elif any([arch in main_arch.keys() for arch in equivariant_arch]):
        irrep.extend(recursive_value_from_key(main_arch, "irrep"))
        symmetries.extend(["Tx", "Ty"])

    # Z2 symmetrization
    if symm_wrapper["Z2"]["on"]:
        irr = 0 if symm_wrapper["Z2"]["trivial_Z2"] else 1
        irrep.append(irr)
        symmetries.append("Z2")

    # print(symmetries, irrep)

    return tuple(symmetries), tuple(irrep)


def irrep_exact_diag(hilbert, H_dense, symmetries, irrep, lattice_size):

    configurations = hilbert.all_states()

    symmetries_config = {
        "Traslation": {
            "on": True if any(symm in ["Tx", "Ty"] for symm in symmetries) else False,
            "subgroup": [0, 1],
        },
        "Z2": {"on": True if "Z2" in symmetries else False},
    }

    group = SymmGroup(
        symmetries_config, lattice_size=lattice_size, all_states=configurations
    )

    # Generate the unitary representations for all symmetry operations
    representations = group.get_unitary_representations(configurations)

    projector = group.get_irrep_projector(representations, irrep)

    basis = sp.linalg.orth(projector)
    block = basis.conj().T @ H_dense @ basis
    evals, evects = sp.linalg.eigh(block)

    E_gr_irrep = evals[0]
    x_ED = basis @ evects[:, 0]

    return float(E_gr_irrep), x_ED


def calc_exact_diag(hilbert, hamiltonian, config_nn, lattice_size):

    N = int(np.prod(lattice_size))
    symmetries, irrep = check_irreps(config_nn)

    if N < 21:
        print("Running exact diagonalization...")
        print("Calculating global ground state...")
        E_gr_global, x_ED_global = sp.sparse.linalg.eigsh(
            hamiltonian.to_sparse(), k=1, return_eigenvectors=True, which="SA"
        )
        E_gr_global = float(E_gr_global)

        if irrep and not N > 12:
            print(f"Calculating {irrep} irrep ground state")
            E_gr_irrep, x_ED_irrep = irrep_exact_diag(
                hilbert, hamiltonian.to_dense(), symmetries, irrep, lattice_size
            )
            E_gr_irrep = float(E_gr_irrep)
        else:
            print("No irrep ground state reference available\n")
            E_gr_irrep, x_ED_irrep = None, None

        irrep = irrep if irrep else None
        symmetries = symmetries if symmetries else None

        return (E_gr_global, x_ED_global), (E_gr_irrep, x_ED_irrep), (symmetries, irrep)

    else:
        return (None, None), (None, None), (symmetries, irrep)
