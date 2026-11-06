from dataclasses import dataclass

import numpy as np
import scipy as sp

from Irreps.SymmBuilder import SymmGroup

equivariant_arch = ["Sequivariant", "SequivariantX"]


@dataclass
class EDResult:
    """Exact-diagonalization reference of one simulation point.

    E_global/x_global: ground state of H (None if the system is too large).
    E_irrep/x_irrep: ground state in the irrep the network is projected to
    (None if there is no irrep or it was not computed).
    """

    E_global: float | None = None
    x_global: np.ndarray | None = None
    E_irrep: float | None = None
    x_irrep: np.ndarray | None = None
    symmetries: list | None = None
    irrep: list | None = None

    @property
    def available(self):
        return self.E_global is not None

    @property
    def E_ref(self):
        """Reference energy for relative errors: irrep if available."""
        return self.E_irrep if self.E_irrep is not None else self.E_global

    @property
    def x_ref(self):
        return self.x_irrep if self.x_irrep is not None else self.x_global

    @property
    def irrep_info(self):
        """{'irrep', 'symmetries'} stored in the results (None without ED/irrep)."""
        if self.available and self.irrep is not None:
            return {"irrep": self.irrep, "symmetries": self.symmetries}
        return None

    def report(self):
        if not self.available:
            return
        print(f"\nEnergy gr_global: {self.E_global:.6f}")
        if self.irrep is not None:
            print(f"Projecting to {self.symmetries} = {self.irrep}")
            if self.E_irrep is not None:
                print(f"Energy gr_irrep: {self.E_irrep:.6f}")
        print("\n")

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

    subgroups = [i for i in [0, 1] if lattice_size[i] > 1]

    symmetries_config = {
        "Traslation": {
            "on": True if any(symm in ["Tx", "Ty"] for symm in symmetries) else False,
            "subgroup": subgroups,
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


def calc_exact_diag(hilbert, hamiltonian, config_nn, lattice_size) -> EDResult:

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

        return EDResult(
            E_global=E_gr_global,
            x_global=x_ED_global,
            E_irrep=E_gr_irrep,
            x_irrep=x_ED_irrep,
            symmetries=symmetries if symmetries else None,
            irrep=irrep if irrep else None,
        )

    return EDResult(
        symmetries=symmetries if symmetries else None, irrep=irrep if irrep else None
    )
