from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


class Sequential(nn.Module):
    """
    Flax module to train module and phase separately
    """

    Seq: Tuple[nn.Module, ...]

    def setup(self):

        self.seq = self.Seq

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for module in self.seq:
            x = module(x)

        return x
