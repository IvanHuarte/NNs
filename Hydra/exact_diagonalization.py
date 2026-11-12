import copy
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


def stage_sectors(config_nn, lattice_size=None):
    """
    (symmetries, irrep) of every stage of the selected Hydra evolution,
    applying the stage 'symm_wrapper' overrides cumulatively (as Hydra does).
    With lattice_size, a Sequivariant backbone with output stride s > 1 and no
    coset only fixes the momentum mod L/s: the translations become "Tx^sx",
    "Ty^sy" (the group the network is actually symmetric under).
    """
    experiment_selection = config_nn["selection"]
    evolution = config_nn[experiment_selection]
    symm_wrapper = copy.deepcopy(config_nn["symm_wrapper"])
    sectors = []
    for stage in sorted(evolution, key=lambda k: int(k[5:])):
        for name, opts in evolution[stage].get("symm_wrapper", {}).items():
            symm_wrapper.setdefault(name, {}).update(opts)
        sectors.append(_sector(evolution[stage]["template"], symm_wrapper))
        if lattice_size is not None and "Tx" in sectors[-1][0]:
            stride = _sequivariant_stride(
                evolution[stage]["template"], config_nn["storage"], symm_wrapper, lattice_size
            )
            if stride != (1, 1):
                symms = tuple(
                    {"Tx": f"Tx^{stride[1]}", "Ty": f"Ty^{stride[0]}"}.get(x, x)
                    for x in sectors[-1][0]
                )
                sectors[-1] = (symms, sectors[-1][1])
    return sectors


def _sequivariant_stride(template, storage, symm_wrapper, lattice_size):
    """(sy, sx) output stride of a Sequivariant backbone without coset, else (1, 1)."""
    import contextlib
    import io

    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    from Hydra.NN.NeuralNetwork import NeuralNetwork
    from Hydra.NN.NN.FluxModules.Sequivariant import Sequivariant
    from Hydra.NN.utils import setup_from_template

    with contextlib.redirect_stdout(io.StringIO()):
        setup = setup_from_template(
            copy.deepcopy(template), copy.deepcopy(storage), copy.deepcopy(symm_wrapper)
        )
        model = NeuralNetwork(setup, lattice_size=lattice_size).model

    def find(m):
        if isinstance(m, Sequivariant):
            return m
        for v in vars(m).values():
            for c in v if isinstance(v, (tuple, list)) else (v,):
                if isinstance(c, nn.Module) and (r := find(c)) is not None:
                    return r
        return None

    seq = find(model)
    if seq is None or (seq.coset is not None and tuple(seq.coset) != (1, 1)):
        return (1, 1)
    backbone = nn.Sequential([m.clone(parent=None) for m in seq.SeqV[:-1]])
    x = jnp.ones((1, *lattice_size, 1))
    out = jax.eval_shape(lambda: backbone.init_with_output(jax.random.PRNGKey(0), x)[0])
    return (lattice_size[0] // out.shape[1], lattice_size[1] // out.shape[2])


def _sector(main_arch, symm_wrapper):
    symmetries = []
    irrep = []

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

    return tuple(symmetries), tuple(irrep)


def check_irreps(config_nn, stage=-1, lattice_size=None):
    """Sector of one stage (default: the last one, whose best state is reported)."""
    return stage_sectors(config_nn, lattice_size)[stage]


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
    # "Ty^s"/"Tx^s": only translations by s (strided backbone without coset),
    # the projector then spans the momenta q + (L/s) a
    T_stride = {x.split("^")[0]: int(x.split("^")[1]) if "^" in x else 1
                for x in symmetries if x.startswith("T")}
    if T_stride:
        for axis, name in ((0, "Ty"), (1, "Tx")):
            q = irrep.pop(0)
            s = T_stride.get(name, 1)
            if lattice_size[axis] // s > 1:
                perm = _translation_perm(all_states, hilbert, lattice_size, axis)
                for _ in range(s - 1):
                    perm = perm[_translation_perm(all_states, hilbert, lattice_size, axis)]
                generators.append((perm, lattice_size[axis] // s, q, "T"))
    if "Z2" in symmetries:
        q = irrep.pop(0)
        if getattr(hilbert, "_total_sz", None) not in (None, 0):
            raise ValueError("Z2 projection needs total_sz = 0 or no Sz constraint.")
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


def calc_exact_diag(hilbert, hamiltonian, config_nn, lattice_size, stage=-1) -> EDResult:
    """ED reference of one stage sector (default: the last stage)."""

    N = int(np.prod(lattice_size))
    symmetries, irrep = check_irreps(config_nn, stage, lattice_size)

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
