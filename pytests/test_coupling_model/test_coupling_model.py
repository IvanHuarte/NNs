"""
CouplingModel package (integrated from ATMOS_VA/VA_project).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_coupling_model -q
"""

import copy
import os
import sys

import jax
import netket as nk
import numpy as np
import pytest
import scipy.sparse.linalg as sla

jax.config.update("jax_enable_x64", True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from CouplingModel import REGISTRED_MODELS, ModelFactory


def j1j2_config(J2, S_operators=False, J2_list=None, fields=(0.0, 0.0, 0.0)):
    block = {"J1": 1.0, "fields": list(fields)}
    if J2_list is None:
        block["J2"] = J2
    else:
        block["J2_list"] = J2_list
    return {
        "mode": "for",
        "kwargs_lattice": {"bc": "periodic", "order": "default_2"},
        "S_operators": S_operators,
        "CM": {"selection": "J1J2Square", "J1J2Square": block},
    }


def ground_energy(H):
    return sla.eigsh(H.to_sparse(), k=1, which="SA")[0][0]


@pytest.mark.parametrize("J2", [0.0, 0.5])
def test_j1j2_matches_netket_heisenberg(J2):
    L = 4
    mf = ModelFactory([L, L], j1j2_config(J2))
    next(mf.get_params())
    cm = mf.get_model().cm
    hi = nk.hilbert.Spin(0.5, N=L * L, total_sz=0)

    E = ground_energy(cm.build_hamiltonian(hi, S_operators=True))
    graph = nk.graph.Square(L, max_neighbor_order=2, pbc=True)
    H_nk = nk.operator.Heisenberg(hi, graph, J=[1.0, J2], sign_rule=[False, False]) / 4
    assert E == pytest.approx(ground_energy(H_nk), abs=1e-10)
    if J2 == 0.0:  # 4x4 Heisenberg, S units
        assert E / L**2 == pytest.approx(-0.70178020, abs=1e-7)


def test_units_and_terms_not_modified():
    # with fields: they scale by 1/2, couplings by 1/4
    mf = ModelFactory([2, 2], j1j2_config(0.3, fields=(0.2, 0.0, 0.3)))
    next(mf.get_params())
    cm = mf.get_model().cm
    before = (np.array(cm.onsite_terms), np.array(cm.coupling_terms))

    H_P = cm.build_hamiltonian(S_operators=False).to_dense()
    H_S = cm.build_hamiltonian(S_operators=True).to_dense()
    cm_c = copy.deepcopy(cm)
    cm_c.onsite_terms = np.asarray(cm.onsite_terms) / 2
    cm_c.coupling_terms = np.asarray(cm.coupling_terms) / 4
    np.testing.assert_allclose(H_S, cm_c.build_hamiltonian().to_dense())
    assert not np.allclose(H_S, H_P / 4)

    # build_hamiltonian does not rescale the model in place (Runner did)
    np.testing.assert_array_equal(cm.onsite_terms, before[0])
    np.testing.assert_array_equal(cm.coupling_terms, before[1])

    E, x = cm.exact_energy_lanczos(k=1, eigenstates=True, S_operators=True)
    assert E[0] == pytest.approx(np.linalg.eigvalsh(H_S)[0])


def test_factory_sweep_and_setup_roundtrip():
    mf = ModelFactory([4, 4], j1j2_config(None, J2_list=[0.4, 0.5]))
    J2s = [params["J2"] for params in mf.get_params()]
    assert J2s == [0.4, 0.5]
    setup = mf.get_setup()
    assert setup["name"] == "J1J2Square" and setup["params"]["J2"] == 0.5
    # saveNload rebuilds the model from the stored setup
    model = ModelFactory.init(setup).get_model()
    assert model.cm.lattice.Q == 16


def test_registered_models():
    assert set(REGISTRED_MODELS) == {
        "Oxalate",
        "LRChain",
        "LRSquare",
        "SSMSquare",
        "J1J2Square",
        "IsingSquare",
    }


def test_invalid_operator():
    mf = ModelFactory([2, 2], j1j2_config(0.0))
    next(mf.get_params())
    cm = mf.get_model().cm
    cm.operators = [["W"], cm.operators[1]]
    with pytest.raises(ValueError, match="operators"):
        cm.build_hamiltonian()
