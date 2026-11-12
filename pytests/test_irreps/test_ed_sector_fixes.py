"""
Regressions: ED sector of a strided Sequivariant without coset, Z2 ED without
Sz constraint, SymmGroup matrix-free projector and OrbitDiagonalizer sanity check.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_irreps -q
"""

import copy
import json
import os
import sys

import jax
import netket as nk
import numpy as np
import pytest
import scipy as sp

jax.config.update("jax_enable_x64", True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from Hydra.exact_diagonalization import calc_exact_diag, irrep_exact_diag, stage_sectors
from Irreps.BasisBuilder import OrbitDiagonalizer
from Irreps.SymmBuilder import SymmGroup


def heisenberg(Ly, Lx, total_sz=0):
    g = nk.graph.Grid([Ly, Lx], pbc=True, max_neighbor_order=2)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes, total_sz=total_sz)
    return hi, nk.operator.Heisenberg(hi, g, J=[1.0, 0.5], sign_rule=[False, False])


def vit_config(token, coset=None):
    cfg = copy.deepcopy(json.load(open(os.path.join(ROOT, "config3_Hydra_NN.json"))))
    cfg["storage"]["ViT2D"].update(token_size=list(token), embedding_d=4, n_heads=1,
                                   n_blocks=1, n_ffn_layers=1)
    cfg["storage"]["OutputHead"]["d_model"] = 4
    seq = {"ViT2D": {}, "OutputHead": {}, "irrep": [2, 2]}
    if coset is not None:
        seq["coset"] = list(coset)
    cfg["selection"] = "T"
    cfg["T"] = {"stage0": {"template": {"Sequivariant": seq}}}
    return cfg


def test_strided_sector_is_min_over_aliased_momenta():
    hi, H = heisenberg(4, 4)
    cfg = vit_config((2, 2))
    assert stage_sectors(cfg, (4, 4)) == [(("Tx^2", "Ty^2"), (2, 2))]
    assert stage_sectors(cfg) == [(("Tx", "Ty"), (2, 2))]  # no size: unchanged
    E = calc_exact_diag(hi, H, cfg, (4, 4)).E_irrep
    E_ref = min(irrep_exact_diag(hi, H, ("Tx", "Ty"), (qy, qx), (4, 4))[0]
                for qy in (0, 2) for qx in (0, 2))
    assert E == pytest.approx(E_ref, abs=1e-8)
    assert E == pytest.approx(-33.831693, abs=1e-5)


def test_coset_keeps_full_lattice_sector():
    cfg = vit_config((2, 2), coset=(2, 2))
    assert stage_sectors(cfg, (4, 4)) == [(("Tx", "Ty"), (2, 2))]


def test_z2_without_sz_constraint():
    g = nk.graph.Grid([2, 3], pbc=True)
    hi = nk.hilbert.Spin(0.5, N=6)
    H = nk.operator.Ising(hi, g, h=1.0)
    evals = []
    for q in (0, 1):
        E, x = irrep_exact_diag(hi, H, ("Tx", "Ty", "Z2"), (0, 0, q), (2, 3))
        flip = np.asarray(hi.states_to_numbers(-np.asarray(hi.all_states())))
        np.testing.assert_allclose(x[flip], (-1) ** q * x, atol=1e-8)
        evals.append(E)
    assert min(evals) == pytest.approx(sp.sparse.linalg.eigsh(H.to_sparse(), k=1, which="SA")[0][0])


def test_symmgroup_projector_action():
    hi, H = heisenberg(2, 3)
    states = hi.all_states()
    setup = {"Traslation": {"on": True, "subgroup": [0, 1]}, "Z2": {"on": True}}
    group = SymmGroup(setup, lattice_size=(2, 3), all_states=states)
    for q in [(0, 0, 0), (1, 2, 1), (0, 1, 0)]:
        P_dense = np.asarray(group.get_irrep_projector(
            group.get_unitary_representations(states), q, norm=True))
        P = group.get_irrep_projector_action(q)
        v = np.random.default_rng(0).standard_normal(states.shape[0]) + 0j
        np.testing.assert_allclose(np.asarray(P(v)), P_dense @ v, atol=1e-10)


def test_orbit_diagonalizer_sanity_check():
    hi, H = heisenberg(2, 3)
    setup = {"Traslation": {"on": True, "subgroup": [0, 1]}, "Z2": {"on": True}}
    group = SymmGroup(setup, lattice_size=(2, 3))
    od = OrbitDiagonalizer(group, hi)
    eigvals, _ = od.diagonalize_by_blocks(H, q_vectors=[(0, 0, 0)], sanity_check=True)
    assert len(eigvals[(0, 0, 0)]) > 0
