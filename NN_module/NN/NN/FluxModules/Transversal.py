from typing import AnyStr, Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp


def final_ensemble(ensem_mode: AnyStr = "sum") -> Callable:

    # Operation selection
    if ensem_mode == "sum":
        return jnp.sum
    elif ensem_mode == "sum_angles":

        def fun(x, axis=0, keepdims=True):
            x = jnp.sum(x, axis=axis, keepdims=keepdims)
            x = (x + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return x

        return fun

    elif ensem_mode == "product":
        return jnp.prod

    elif ensem_mode == "mean":
        return jnp.mean
    elif ensem_mode == "modulus_marshall":

        def fun(x, axis=0, keepdims=True):
            x = jnp.log(x[1] * jnp.exp(x[0]))
            return x.astype(jnp.complex128)

        return fun

    elif "None":

        def lambda_wrap(x, axis=0, keepdims=True):
            return x

        return lambda_wrap

    else:
        raise ValueError(f"Ensemble mode {ensem_mode} not recognized.")


class Transversal(nn.Module):
    """
    Flax module to train module and phase separately
    """

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        B = x.shape[0]
        # print(f"x_in: {x.shape}")

        x = jnp.stack([module(x) for module in self.Trans], axis=0)
        # print(f"res: {x.shape}")

        # Norm
        if self.post_norm:
            x = nn.LayerNorm()(x.swapaxes(0, -1)).swapaxes(0, -1)

        # print(f"x_norm: {x.shape}")

        x = final_ensemble(ensem_mode=self.operation)(x, axis=0, keepdims=True)  ######
        # jax.debug.print("x_after: {} \n\n", x)

        # print(f"final_ensemble: {x.shape}")
        x = jnp.atleast_2d(x).reshape(B, *x.shape[2:])

        # print(f"x_group: {x.shape}")
        # print(f"\n")
        return x
