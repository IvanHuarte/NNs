import jax.numpy as jnp


def linear(epochs, y0, y1, steps=None):
    x = jnp.arange(epochs)
    
    if steps is None:
        return y0 + (y1-y0)/epochs * x

    k = (x * steps) // epochs
    k = jnp.minimum(k, steps - 1)
    return y0 + (y1 - y0) * (k / steps)

def exponential(epochs, y0, y1, steps=None):
    x = jnp.arange(epochs)

    if steps is None:
        # decay continuo
        return y0 * (y1 / y0) ** (x / epochs)

    k = (x * steps) // epochs
    k = jnp.minimum(k, steps - 1)

    return y0 * (y1 / y0) ** (k / steps)