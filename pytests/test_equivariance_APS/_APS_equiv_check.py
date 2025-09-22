import jax
import jax.numpy as jnp
from NN_module.models.CvT_APS import get_maxnorm_indices

def APS_equiv_check(x0, params, model, atol=1e-5, v=0):

    success = True

    B0, H0, W0, _ = x0.shape
    shifts_lat0 = jnp.array([(i, j) for i in range(H0) for j in range(W0)])

    # y0 = model.apply(params, x0)
    y0, P0 = model.apply(params, x0, return_shifts=True)

    B1, H1, W1, C1 = y0.shape
    stride=tuple([H0 // H1, W0 // W1])

    assert B0 == B1, f"Batch dimension must be equal before and after ffn....strange"
    shifts_lat1 = jnp.array([(i*stride[0], j*stride[1]) for i in range(H1) for j in range(W1)])

    y_check_list = jnp.array(
        [jnp.roll(y0, shift, axis=(1, 2)) for shift in shifts_lat1]
    )

    for shift in shifts_lat0:

        x_roll = jnp.roll(x0, shift=shift, axis=(1, 2))
        # y_roll = model.apply(params, x_roll)
        y_roll, P_roll = model.apply(params, x_roll, return_shifts=True)

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


