from .Traslations import Traslation, TraslationAnchor
from .Z2 import Z2

### Wrapper over other symmetry modules. It takes care of the setup and call methods,
# allowing the user to focus on the specific implementation of the symmetry module.

import jax
import jax.numpy as jnp
import flax.linen as nn


class SymmWrapper(nn.Module):
    """
    A wrapper for symmetry-based modules. It takes care of the setup and call methods, allowing the user to focus on the specific implementation of the symmetry module.
    """

    SymmModule: nn.Module

    def setup(self):
        self.symm_module = self.SymmModule

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.symm_module(x)
