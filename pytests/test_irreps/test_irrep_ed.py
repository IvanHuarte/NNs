"""Matrix-free irrep_exact_diag against the dense SymmGroup projector."""
import netket as nk
import numpy as np
import pytest
import scipy as sp

from Irreps.SymmBuilder import SymmGroup
from Hydra.exact_diagonalization import irrep_exact_diag


def dense_irrep_ed(hi, H, symmetries, irrep, lattice_size):
    """Former dense implementation (reference)."""
    states = hi.all_states()
    setup = {
        "Traslation": {"on": any(s in ("Tx", "Ty") for s in symmetries),
                       "subgroup": [i for i in (0, 1) if lattice_size[i] > 1]},
        "Z2": {"on": "Z2" in symmetries},
    }
    group = SymmGroup(setup, lattice_size=lattice_size, all_states=states)
    P = group.get_irrep_projector(group.get_unitary_representations(states), irrep)
    basis = sp.linalg.orth(np.asarray(P))
    evals, evects = sp.linalg.eigh(basis.conj().T @ H.to_dense() @ basis)
    return evals[0], basis @ evects[:, 0]


def model(Ly, Lx):
    g = nk.graph.Grid([Ly, Lx], pbc=True, max_neighbor_order=2)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes, total_sz=0)
    return hi, nk.operator.Heisenberg(hi, g, J=[1.0, 0.5], sign_rule=[False, False])


@pytest.mark.parametrize("size", [(2, 3), (2, 4)])
@pytest.mark.parametrize("with_z2", [False, True])
def test_matches_dense(size, with_z2):
    hi, H = model(*size)
    symm = ("Tx", "Ty", "Z2") if with_z2 else ("Tx", "Ty")
    for qy in range(size[0]):
        for qx in range(size[1]):
            for qz in ((0, 1) if with_z2 else (None,)):
                irrep = (qy, qx) + (() if qz is None else (qz,))
                E_ref, x_ref = dense_irrep_ed(hi, H, symm, irrep, size)
                E, x = irrep_exact_diag(hi, H, symm, irrep, size)
                assert E == pytest.approx(E_ref, abs=1e-8)
                # eigenvector: same state up to a phase (skip degenerate levels)
                assert np.linalg.norm(x) == pytest.approx(1.0)
                if abs(np.vdot(x_ref, x)) < 0.999:
                    Hx = H.to_sparse() @ x
                    assert np.linalg.norm(Hx - E * x) < 1e-6


def test_z2_needs_sz0():
    g = nk.graph.Grid([2, 3], pbc=True)
    hi = nk.hilbert.Spin(0.5, N=6, total_sz=1)
    H = nk.operator.Heisenberg(hi, g)
    with pytest.raises(ValueError):
        irrep_exact_diag(hi, H, ("Tx", "Ty", "Z2"), (0, 0, 0), (2, 3))
