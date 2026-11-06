"""
Sequivariant options: phase_mode ("angle" legacy / "log") and coset (exact
momentum on the full lattice for backbones with stride/tokens > 1), plus
Hydra.resolve_load_path (directory / glob in load_from).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_irreps -q
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from NN_module.NN.Hydra import resolve_load_path
from NN_module.NN.NN.FluxModules.Sequivariant import Sequivariant
from NN_module.NN.NN.SingleModules.FluxFunc import OutputHead
from NN_module.NN.NN.SingleModules.ViT2D import ViT2D

L = 8


def model(token, irrep, **kw):
    vit = ViT2D(token_size=token, embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1)
    return Sequivariant(SeqV=(vit, OutputHead(d_model=8)), irrep=irrep, **kw)


def configs(n=5):
    x = np.tile(np.array([1.0, -1.0] * (L * L // 2)), (n, 1))
    rng = np.random.default_rng(0)
    for row in x:
        rng.shuffle(row)
    return jnp.asarray(x.reshape(n, L, L, 1))


def momentum_phases(m, params, x, shift, axis):
    d = m.apply(params, jnp.roll(x, shift, axis=axis)) - m.apply(params, x)
    assert np.abs(np.asarray(d.real)).max() < 1e-10, "modulus not invariant"
    return np.angle(np.exp(1j * np.asarray(d.imag))).ravel() / (2 * np.pi / L)


def same_phase(a, b):
    return np.allclose(np.exp(2j * np.pi * a / L), np.exp(2j * np.pi * b / L), atol=1e-8)


@pytest.mark.parametrize("irrep", [(0, 0), (4, 4), (3, 1), (2, 0)])
def test_coset_gives_full_lattice_momentum(irrep):
    m = model((2, 2), irrep, coset=(2, 2), phase_mode="log")
    x = configs()
    p = m.init(jax.random.key(0), x)
    for axis, n in ((1, irrep[0]), (2, irrep[1])):
        assert same_phase(momentum_phases(m, p, x, 1, axis), n)


def test_without_coset_only_token_translations():
    """Tokens 2x2: eigenstate of T_2 only (momenta k and k+pi mixed)."""
    m = model((2, 2), (1, 0))
    x = configs()
    p = m.init(jax.random.key(0), x)
    assert same_phase(momentum_phases(m, p, x, 2, 1), 2)  # T_2: n=1 on 8 sites
    with pytest.raises(AssertionError):
        momentum_phases(m, p, x, 1, 1)


def test_log_mode_same_phase_and_params_as_angle():
    x = configs()
    ma, ml = model((1, 1), (1, 2)), model((1, 1), (1, 2), phase_mode="log")
    p = ma.init(jax.random.key(0), x)
    assert jax.tree.map(jnp.shape, p) == jax.tree.map(jnp.shape, ml.init(jax.random.key(0), x))
    la, ll = ma.apply(p, x), ml.apply(p, x)
    # psi_log = psi_angle * |z|: same phase, different modulus
    np.testing.assert_allclose(np.exp(1j * (ll.imag - la.imag)), 1.0, atol=1e-8)
    assert same_phase(momentum_phases(ml, p, x, 1, 2), 2)


def test_bad_phase_mode():
    x = configs(1)
    with pytest.raises(ValueError):
        model((1, 1), (0, 0), phase_mode="foo").init(jax.random.key(0), x)


def test_resolve_load_path(tmp_path):
    f = tmp_path / "a.json"
    assert resolve_load_path(str(f)) == str(f)  # plain paths untouched
    d = tmp_path / "run" / "x" / "UUID_1"
    d.mkdir(parents=True)
    old, new = d / "M_results_old.json", d / "M_results_new.json"
    old.write_text("{}")
    time.sleep(0.01)
    new.write_text("{}")
    os.utime(old, (1, 1))
    assert resolve_load_path(str(tmp_path / "run")) == str(new)
    assert resolve_load_path(str(tmp_path / "run" / "**" / "*old.json")) == str(old)
    with pytest.raises(FileNotFoundError):
        resolve_load_path(str(tmp_path / "empty*"))
