import jax
import jax.numpy as jnp
import netket as nk


def get_ST_schedule(name, setup):
    """
    Returns a learning rate schedule as an array whose elements are a constant
    learning rate in each segment.

    """

    if name == "segment":
        return jnp.array(setup["lr"])

    elif name == "stairs_schedule":
        lr_schedule = jnp.logspace(
            start=jnp.log10(setup["lr0"]),
            stop=jnp.log10(setup["lr_min"]),
            num=setup["total_segments"],
        )
        return lr_schedule
