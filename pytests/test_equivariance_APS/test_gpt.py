import pytest
import jax
import jax.numpy as jnp
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.models.CvTaps import ConvAPS
from pytests.test_equivariance_APS._APS_equiv_check import APS_equiv_check


from NN_module.models.CvTaps import Conv_APS, APS_equivariance_adapter  # ajusta import


def polyphase_argmax(x, stride):
    """
    Calcula la norma de cada polifase y devuelve (p,q) y la matriz de normas.
    Asume x en formato NHWC.
    """
    B, H, W, C = x.shape
    s_h, s_w = stride
    norms = []
    for p in range(s_h):
        for q in range(s_w):
            comp = x[:, p::s_h, q::s_w, :]   # polifase
            n = jnp.sum(jnp.square(comp))
            norms.append(n)
    norms = jnp.array(norms)
    idx = int(jnp.argmax(norms))
    p = idx // s_w
    q = idx % s_w
    return (p, q, norms.reshape((s_h, s_w)))


def P_anchor(x, stride):
    """
    Implementación directa del operador P (polyphase anchoring).
    """
    p, q, norms = polyphase_argmax(x, stride)
    x_hat = jnp.roll(x, shift=(-p, -q), axis=(1, 2))
    return x_hat, (p, q), norms


def test_debug_equivariance():
    key = jax.random.PRNGKey(int(time.time()))
    lattice_size = (8, 8)
    strides = (2, 2)
    C_in, C_out = 3, 3

    model = Conv_APS(
        channels=C_out,
        strides=strides,
        padding="CIRCULAR",
    )

    x0_shape = (1, *lattice_size, C_in)
    x0 = jax.random.choice(key, jnp.array([-1, 1]), shape=x0_shape)
    params = model.init(key, x0)

    # salida base
    y0 = model.apply(params, x0)

    pX, qX, _ = polyphase_argmax(x0, strides)
    P_x, _, _ = P_anchor(x0, strides)

    H0, W0 = lattice_size
    atol = 1e-8
    ok = True

    for i in range(H0):
        for j in range(W0):
            shift = (i, j)
            x_roll = jnp.roll(x0, shift=shift, axis=(1, 2))
            y_roll = model.apply(params, x_roll)

            # polifase del input desplazado
            pGX, qGX, _ = polyphase_argmax(x_roll, strides)

            # g' = Delta - pGX + pX
            gprime_h = shift[0] - pGX + pX
            gprime_w = shift[1] - qGX + qX

            divisible = (gprime_h % strides[0] == 0) and (gprime_w % strides[1] == 0)
            out_shift = (int(gprime_h // strides[0]), int(gprime_w // strides[1]))

            y_expected = jnp.roll(y0, shift=out_shift, axis=(1, 2))
            same = jnp.allclose(y_expected, y_roll, atol=atol)

            if not (divisible and same):
                ok = False
                print("❌ FALLA para shift=", shift)
                print("  pX,qX:", (int(pX), int(qX)), " pGX,qGX:", (int(pGX), int(qGX)))
                print("  g' (h,w):", (int(gprime_h), int(gprime_w)),
                      " divisible_by_stride:", divisible)
                print("  out_shift en salida:", out_shift)
                print("  max abs diff:", float(jnp.max(jnp.abs(y_expected - y_roll))))
                print("---")
            else:
                print("✅ PASA para shift=", shift, " -> out_shift:", out_shift)

    assert ok, "Alguna traslación rompe la equivarianza"