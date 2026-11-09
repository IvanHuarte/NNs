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
        # Autoregressive models work on the flattened (B, N) configuration,
        # in the sampling order. Accepts (N,), (B, N) or (B, Lx, Ly, 1).
        x = x.reshape(-1, self.hilbert.size)

        x = self.ARNN_model(x)

        return jnp.atleast_1d(jnp.squeeze(x))

    @nn.compact
    def conditionals_log_psi(self, inputs):
        return self.ARNN_model.conditionals_log_psi(inputs)

    def conditional(self, inputs: jnp.ndarray, index: int) -> jnp.ndarray:
        # Same as NetKet's AbstractARNN.conditional, redefined so that
        # ARDirectSampler does not depend on the installed NetKet version.
        return self.conditionals(inputs)[:, index, :]
