import jax
import jax.numpy as jnp
import flax.linen as nn


class Z2(nn.Module):

    Z2Wrap: nn.Module
    trivial_Z2: bool = True

    def setup(self):
        self.worker = self.Z2Wrap

    def __call__(self, x):

        output_x = jnp.atleast_1d(self.worker(x))
        output_inv_x = jnp.atleast_1d(self.worker(-x))

        # Concatenamos las dos contribuciones
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=0)
            return res
        else:
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)
            return res
