import jax.numpy as jnp
import flax.linen as nn


# Wrapper over other symmetry modules.

from .Traslations import Traslation, TraslationAnchor
from .Z2 import Z2


class SymmWrapper(nn.Module):
    """
    A wrapper for symmetry-based modules. It takes care of the setup and call methods, allowing the user to focus on the specific implementation of the symmetry module.
    """

    SymmModel: nn.Module

    def setup(self):
        self.NN_model = self.SymmModel

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)

        x = self.NN_model(x)

        return jnp.squeeze(x)
