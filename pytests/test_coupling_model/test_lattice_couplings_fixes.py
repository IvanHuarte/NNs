"""
Regressions: duplicated neighbours on L=2 axes (np.add.at) and minimum-image
distances on non-square triangular lattices.

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_coupling_model -q
"""

import itertools
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
from CouplingModel import ModelFactory
from CouplingModel.lattice.lattice import Triangular


def brute_force_j1j2(Ly, Lx, J2):
    """Pauli sigma.sigma on every directed bond / 2: periodic images counted separately."""
    hi = nk.hilbert.Spin(0.5, N=Ly * Lx, total_sz=0)
    sx, sy, sz = nk.operator.spin.sigmax, nk.operator.spin.sigmay, nk.operator.spin.sigmaz
    H = 0.0
    for y, x in itertools.product(range(Ly), range(Lx)):
        i = y * Lx + x
        for (dy, dx), J in [((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), J2), ((1, -1), J2)]:
            j = ((y + dy) % Ly) * Lx + (x + dx) % Lx
            for s in (sx, sy, sz):
                H = H + J * s(hi, i) @ s(hi, j)
    return hi, H


@pytest.mark.parametrize("size", [(2, 2), (4, 2), (2, 4), (3, 2)])
def test_j1j2_L2_matches_brute_force(size):
    J2 = 0.5
    config = {
        "mode": "for",
        "kwargs_lattice": {"bc": "periodic", "order": "default_2"},
        "S_operators": False,
        "CM": {"selection": "J1J2Square",
               "J1J2Square": {"J1": 1.0, "J2": J2, "fields": [0.0, 0.0, 0.0]}},
    }
    mf = ModelFactory(list(size), config)
    next(mf.get_params())
    hi, H_ref = brute_force_j1j2(*size, J2)
    H = mf.get_model().cm.build_hamiltonian(hi, S_operators=False)
    E = sla.eigsh(H.to_sparse(), k=1, which="SA")[0][0]
    E_ref = sla.eigsh(H_ref.to_sparse(), k=1, which="SA")[0][0]
    assert E == pytest.approx(E_ref, abs=1e-10)
    if size in ((2, 2), (4, 2)):
        assert E == pytest.approx({(2, 2): -12.0, (4, 2): -24.0}[size], abs=1e-8)


@pytest.mark.parametrize("L", [(4, 2), (2, 4), (3, 6), (6, 3), (4, 3)])
def test_triangular_minimum_image(L):
    lat = Triangular(L[0], L[1], bc="periodic")
    pos = np.asarray(lat.positions)
    Lu1, Lu2 = L[0] * np.asarray(lat.basis[0]), L[1] * np.asarray(lat.basis[1])
    images = [n1 * Lu1 + n2 * Lu2 for n1 in range(-3, 4) for n2 in range(-3, 4)]
    d = pos[:, None, :] - pos[None, :, :]
    ref = np.min([np.linalg.norm(d + im, axis=-1) for im in images], axis=0)
    np.testing.assert_allclose(lat.calc_distances(pos, "periodic"), ref, atol=1e-9)
