import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D
from NN_module.NN.toolbox import CarreteSign, AddPhase


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
