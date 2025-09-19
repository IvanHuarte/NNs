import jax
import jax.numpy as jnp
from NN_module.models.CvT_APS import get_maxnorm_indices
import itertools


def get_channel_indices(H, W, C):
    assert (H * W) ** C < 3e5, f"Too much channels or lattice size"
    grid_tras_ind = jnp.array([(i, j) for i in range(H) for j in range(W)])
    combo_idx = jnp.array(list(itertools.product(range(H * W), repeat=C)))
    return grid_tras_ind[combo_idx]


def get_checklist_from_indices(y, indices):

    y_t = y.transpose((0, 3, 1, 2))

    def roll_combination(_, shift_comb):

        def rollit(c, shift):
            ys = jnp.roll(y_t[0, c], shift=shift, axis=(-2, -1))
            return c + 1, ys

        _, y_comb = jax.lax.scan(rollit, 0, shift_comb)
        return 0, y_comb

    _, y_checklist = jax.lax.scan(roll_combination, 0, indices)

    return y_checklist.transpose((0, 2, 3, 1))


def APS_equiv_check(x0, params, model, atol=1e-5, v=0):

    success = True

    B0, H0, W0, _ = x0.shape
    shifts_lat0 = jnp.array([(i, j) for i in range(H0) for j in range(W0)])

    y0 = model.apply(params, x0)

    B1, H1, W1, C1 = y0.shape
    assert B0 == B1, f"Batch dimension must be equal before and after ffn....strange"
    shifts_lat1 = jnp.array([(i * H1, j * W1) for i in range(H0) for j in range(W0)])
    y_check_list = jnp.array(
        [jnp.roll(y0, shift, axis=(1, 2)) for shift in shifts_lat1]
    )

    for shift in shifts_lat0:

        x_roll = jnp.roll(x0, shift=shift, axis=(1, 2))
        y_roll = model.apply(params, x_roll)

        check = []
        for y_check in y_check_list:
            check.append(jnp.allclose(y_check, y_roll, atol=atol))

        if not any(check):
            success = False
            if v > 0:
                print(f"Equivarianza fallida para shift={shift}")
        else:
            if v > 0:
                print(f"Equivarianza existente para shift={shift}")

    if success:
        print("✅ El modelo es equivariante!")
    else:
        print("❌ Alguna operación rompe la equivarianza")
