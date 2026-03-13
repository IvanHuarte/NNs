import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D
from NN_module.NN.toolbox import CarreteSign, AddPhase


def get_op_function(mode):

    if mode is None:
        return lambda x: x

    elif mode == "modulus":

        def constant_phase(x):
            phase = jnp.zeros_like(x)
            return x.astype(dtype=jnp.complex128) + 1j * phase.astype(
                dtype=jnp.complex128
            )

        return constant_phase

    elif mode == "phase":

        def constant_phase(x):
            modulus = jnp.ones_like(x)
            return modulus.astype(dtype=jnp.complex128) + 1j * x.astype(
                dtype=jnp.complex128
            )

        return constant_phase

    else:
        raise ValueError(f"Mode {mode} not recognized.")


class SingleModule(nn.Module):
    """
    Flax module to train modulus and phase separately
    """

    Single: nn.Module
    mode: str = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = self.Single(x)

        x = get_op_function(self.mode)(x)

        return x
