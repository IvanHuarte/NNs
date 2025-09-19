import jax.numpy as jnp


def equivariance_traslation_test(x0, lattice_size, params, model, atol=1e-5, v=0):
    """
    Testea la equivarianza traslacional de un modelo sobre una entrada dada.

    Esta función verifica que la arquitectura `model` cumpla con la propiedad de
    equivarianza traslacional. Para cada traslación posible en la retícula
    (definida por `lattice_size`), compara la salida del modelo cuando se aplica
    la traslación al input (`T(x)`) con la traslación aplicada a la salida original
    (`T(f(x))`). Si el modelo es perfectamente equivariante, ambas deberían coincidir
    dentro de una tolerancia numérica.

    Parámetros
    ----------
    x0 : jax.numpy.ndarray
        Entrada original al modelo. Se asume que tiene forma compatible con
        `lattice_size` (por ejemplo, `(batch, N)` o `N` aplanado).
    lattice_size : tuple of int
        Tamaño de la retícula (ej. `(Lx, Ly)` para una red 2D).
    params : PyTree
        Parámetros del modelo, normalmente obtenidos con `model.init(...)`.
    model : flax.linen.Module
        El modelo a evaluar. Debe tener el método `apply(params, x)`.
    atol : float, opcional
        Tolerancia absoluta para la comparación numérica. Por defecto `1e-7`.

    Raises
    ------
    AssertionError
        Si alguna traslación no cumple la condición de equivarianza dentro de
        la tolerancia especificada.

    Notas
    -----
    - Este test asume que la salida del modelo tiene la misma estructura espacial
      que la entrada para poder aplicar `jnp.roll` en los mismos ejes.
    - Imprime para cada traslación el error cuadrático
      `|T_i(f(x)) - f(T_i(x))|²`.
    """
    success = True
    shifts = [(i, j) for i in range(lattice_size[0]) for j in range(lattice_size[1])]
    y0 = model.apply(params, x0)
    out_shape = y0.shape

    for i, shift in enumerate(shifts):

        x_roll = jnp.roll(x0, shift=shift, axis=(1, 2))
        y_roll = model.apply(params, x_roll)
        y0_shift = jnp.roll(y0.reshape(*out_shape), shift=shift, axis=(1, 2))

        if not jnp.allclose(y0_shift, y_roll, atol=atol):
            print(f"Equivarianza fallida para shift={shift}")
            success = False

        if v > 0:
            print(f"Traslation #{i}: {shift}")
            print(f"|T_i(f(x)) - f(T_i(x))|² = {jnp.sum((y0_shift-y_roll)**2)}\n\n")

    if success:
        print("✅ El modelo es equivariante!")
    else:
        print("❌ Alguna operación rompe la equivarianza")


def equivariance_traslation_all_test(x0, lattice_size, params, model, atol=1e-5, v=0):
    """
    Test translational equivariance of a 2D lattice model.

    This test checks whether the model `model` with parameters `params` is
    equivariant under translations along the spatial dimensions of the input `x0`.
    Translational equivariance means that applying a shift to the input produces
    the same result as applying the corresponding shift to the output.

    Parameters
    ----------
    x0 : jnp.ndarray
        Input array defined on a 2D lattice.
    lattice_size : tuple of int
        Size of the lattice (n_rows, n_cols) defining the possible shifts.
    params : dict
        Model parameters (e.g., neural network weights).
    model : flax.linen.Module
        Model to be tested. Must implement `.apply(params, x)`.
    atol : float, optional (default=1e-5)
        Absolute tolerance used by `jnp.allclose` when comparing outputs.
    v : int, optional (default=0)
        Verbosity level. If v>0, prints confirmation for valid shifts during the test.

    Returns
    -------
    None
        Prints the test results:
        - A warning for each translation that breaks equivariance.
        - A final message confirming whether the model is equivariant.
    """

    success = True
    shifts = [(i, j) for i in range(lattice_size[0]) for j in range(lattice_size[1])]
    y0 = model.apply(params, x0)
    out_shape = y0.shape
    y_check_list = jnp.array(
        [
            jnp.roll(y0, (i, j), axis=(1, 2))
            for i in range(out_shape[1])
            for j in range(out_shape[2])
        ]
    )

    for shift in shifts:

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
