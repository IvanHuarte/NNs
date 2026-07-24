import optax
import jax.numpy as jnp

def transformation_dictionary(optimizer, modes, lr_func):

    transformation = {"freeze": optax.set_to_zero()}

    for branch, lr_f in zip(modes, lr_func):
        transformation[f"train_{branch}"] = optax.chain(
            # optax.clip(1.0),
            optimizer(lr_f),
        )

    return transformation
