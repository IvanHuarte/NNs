from dataclasses import dataclass

import numpy as np
import scipy as sp


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


def _translation_perm(all_states, hilbert, lattice_size, axis):
    """Index permutation p of a one-site translation: U|s_i> = |s_p[i]>, U|s> = |T s>."""
    grid = np.asarray(all_states).reshape(-1, *lattice_size)
    shifted = np.roll(grid, 1, axis=axis + 1).reshape(len(all_states), -1)
    return np.asarray(hilbert.states_to_numbers(shifted))


def irrep_exact_diag(hilbert, hamiltonian, symmetries, irrep, lattice_size, tol=1e-10):
    """
    Lowest eigenpair of H inside the irrep, matrix-free.

    Same convention as the former dense version (Irreps.SymmGroup):
        P = 1/|G| sum_n chi(q, n) U^n,   U|s> = |T s>,
        chi_T(q, n) = exp(+2 pi i q n / L),   chi_Z2(q, n) = (-1)^(q n),
    irrep = (q_y, q_x[, q_Z2]) with the translations ordered as lattice axes (0, 1).

    Symmetries act as index permutations on the basis (never as matrices) and
    the eigenpair comes from Lanczos on  A = H P + c (1 - P):  H commutes with P,
    so A is H on Im(P) and c on ker(P); c above the spectrum of H pushes every
    state outside the irrep out of the low end. Memory: a few vectors plus sparse H.
    """
    H = sp.sparse.csr_matrix(hamiltonian.to_sparse(), copy=True)  # netket arrays are read-only
    all_states = np.asarray(hilbert.all_states())
    dim = H.shape[0]

    irrep_label = tuple(irrep)
    irrep = list(irrep)
    # (permutation, order, q) per generator, in the order of `irrep`
    generators = []
    if any(symm in ("Tx", "Ty") for symm in symmetries):
        for axis in (0, 1):
            q = irrep.pop(0)
            if lattice_size[axis] > 1:
                perm = _translation_perm(all_states, hilbert, lattice_size, axis)
                generators.append((perm, lattice_size[axis], q, "T"))
    if "Z2" in symmetries:
        q = irrep.pop(0)
        if not np.allclose(all_states.sum(-1), 0):
            raise ValueError("Z2 projection needs the Sz = 0 sector (total_sz = 0).")
        perm = np.asarray(hilbert.states_to_numbers(-all_states))
        generators.append((perm, 2, q, "Z2"))

    def char(kind, q, n, order):
        return (-1.0) ** (q * n) if kind == "Z2" else np.exp(2j * np.pi * q * n / order)

    def apply_U(v, perm):  # (U v)[perm[i]] = v[i]
        w = np.empty_like(v)
        w[perm] = v
        return w

    def P(v):
        # The group is abelian and a direct product: P = prod_g P_g
        for perm, order, q, kind in generators:
            acc = np.zeros_like(v)
            u = v
            for n in range(order):
                acc += char(kind, q, n, order) * u
                u = apply_U(u, perm)
            v = acc / order
        return v

    # Gershgorin bound: c above the whole spectrum of H
    c = float(abs(H).sum(axis=1).max()) + 1.0

    def matvec(v):
        v = np.asarray(v, dtype=complex).ravel()
        pv = P(v)
        return H @ pv + c * (v - pv)

    A = sp.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, dtype=complex)
    v0 = P(np.random.default_rng(0).standard_normal(dim) + 0j)
    evals, evects = sp.sparse.linalg.eigsh(A, k=1, which="SA", tol=tol, v0=v0)

    E_gr_irrep = float(evals[0])
    if E_gr_irrep >= c - 1e-8:
        raise ValueError(f"Irrep {irrep_label} is empty in this Hilbert space.")

    x_ED = P(evects[:, 0])
    x_ED /= np.linalg.norm(x_ED)

    return E_gr_irrep, x_ED


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

        if irrep:
            print(f"Calculating {irrep} irrep ground state")
            E_gr_irrep, x_ED_irrep = irrep_exact_diag(
                hilbert, hamiltonian, symmetries, irrep, lattice_size
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
