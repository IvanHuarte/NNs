import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# Wrapper over other symmetry modules.

from .Traslations import Traslation
from .Z2 import Z2


class SymmWrapper(nn.Module):
    """
    A wrapper for symmetry-based modules. It takes care of the setup and call methods, allowing the user to focus on the specific implementation of the symmetry module.
    """
    SymmModel: nn.Module
    lattice_size: Tuple[int, int]

    def setup(self):
        self.NN_model = self.SymmModel

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        x = x.reshape(x.shape[0], *self.lattice_size, 1)

        x = self.NN_model(x)

        return jnp.atleast_1d(jnp.squeeze(x))
