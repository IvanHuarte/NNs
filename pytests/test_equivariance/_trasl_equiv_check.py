import jax.numpy as jnp

from NN_module.NN_utils import Translations2D


def traslation_equivariant(x0, params, model, atol=1e-7, v=0):
    success = True

    # x0 = (B, H, W, Ch)
    _, Hx, Wx, _ = x0.shape

    y0 = model.apply(params, x0)
    _, Hy, Wy, _ = y0.shape

    stride = (Hx // Hy, Wx // Wy)

    x0_translations = (
        Translations2D(x0, stride=stride)
        .transpose((1, 2, 0, 3, 4, 5))
        .reshape(Hy * Wy, *x0.shape)
    )
    y0_translations = (
        Translations2D(y0, stride=(1, 1))
        .transpose((1, 2, 0, 3, 4, 5))
        .reshape(Hy * Wy, *y0.shape)
    )

    idx = jnp.unravel_index(jnp.arange(Hx * Wx), (Hx, Wx))
    shift_idx = jnp.array(idx).T
    mask = (shift_idx[:, 0] % stride[0] == 0) & (shift_idx[:, 1] % stride[1] == 0)
    shift_idx = shift_idx[mask]

    for shift, x_roll, y0_roll in zip(shift_idx, x0_translations, y0_translations):
        y_roll = model.apply(params, x_roll)

        if not jnp.allclose(y0_roll, y_roll, atol=atol):
            print(f"Equivarianza fallida para shift={shift}")
            success = False

        if v > 0:
            print(f"Traslation {shift}:")
            print(f"|T_i(f(x)) - f(T_i(x))|² = {jnp.sum((y0_roll-y_roll)**2)}\n\n")
            # print(f"y0_shift:\n {y0_shift.reshape(1, -1, 4, 4)}\n")
            # print(f"y_roll:\n {y_roll.reshape(1, -1, 4, 4)}\n")

    if success:
        print("✅ El modelo es equivariante!")
    else:
        print("❌ Alguna operación rompe la equivarianza")
