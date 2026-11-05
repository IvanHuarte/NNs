import flax.linen as nn
import jax.numpy as jnp
import netket as nk


class ARNNWrapper(nk.models.AbstractARNN):
    """
    A wrapper for symmetry-based modules. It takes care of the setup and call methods, allowing the user to focus on the specific implementation of the symmetry module.
    """

    ARNN: nn.Module
    hilbert: nk.hilbert
    lattice_size: tuple[int, int]
    machine_pow: int = 2

    def setup(self):
        self.ARNN_model = self.ARNN

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        x = x.reshape(x.shape[0], *self.lattice_size, 1)

        x = self.ARNN_model(x)

        return jnp.atleast_1d(jnp.squeeze(x))

    @nn.compact
    def conditionals_log_psi(self, inputs):
        return self.ARNN_model.conditionals_log_psi(inputs)
