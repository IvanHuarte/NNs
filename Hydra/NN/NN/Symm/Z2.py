import flax.linen as nn
import jax
import jax.numpy as jnp


class Z2Standard(nn.Module):
    Z2Wrap: nn.Module
    trivial_Z2: bool = True

    def setup(self):
        self.worker = self.Z2Wrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        output_x = self.worker(x)
        output_inv_x = self.worker(-x)

        # Concatenamos las dos contribuciones
        z2_stack = jnp.stack([output_x, output_inv_x], axis=1)
        if not self.trivial_Z2:
            # complex so log of a negative sum is finite; works for (B,) and (B,1) heads
            z2_stack = z2_stack.astype(jnp.result_type(z2_stack.dtype, jnp.complex64))

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=1)
            return res
        else:
            b = jnp.array([1.0, -1.0]).reshape((1, 2) + (1,) * (output_x.ndim - 1))
            res = jax.nn.logsumexp(z2_stack, b=b, axis=1)
            return res


class Z2SymmModulus(nn.Module):
    Z2Wrap: nn.Module

    def setup(self):
        self.worker = self.Z2Wrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        out_x = jnp.atleast_1d(self.worker(x))
        out_inv_x = jnp.atleast_1d(self.worker(-x))

        symm_log_modulus = jnp.stack([out_x.real, out_inv_x.real], axis=1).mean(axis=1)
        phase = out_x.imag

        return symm_log_modulus + 1.0j * phase


class Z2(nn.Module):

    Z2Wrap: nn.Module
    trivial_Z2: bool = True
    symm_modulus: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_modulus:
            x = Z2SymmModulus(Z2Wrap=self.Z2Wrap)(x)
        else:
            x = Z2Standard(
                Z2Wrap=self.Z2Wrap,
                trivial_Z2=self.trivial_Z2,
            )(x)

        return x
