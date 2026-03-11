import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D
from NN_module.NN.toolbox import CarreteSign, AddPhase


class Traslation(nn.Module):

    lattice_size: Tuple[int, int]
    TWrap: nn.Module

    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    save_memory: bool = False

    squeeze: Callable = lambda x: x

    def setup(self):

        self.wrap = self.Twrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        worker = self.wrap(x)

        na, nb = jnp.unravel_index(
            jnp.arange(self.lattice_size[0] * self.lattice_size[1]), self.lattice_size
        )

        characters = jnp.exp(
            2j
            * jnp.pi
            * (
                self.irrep[0] * na / self.lattice_size[0]
                + self.irrep[1] * nb / self.lattice_size[1]
            )
        )

        # 2D traslation
        traslational_x = traslations_2D(
            x,
            size=self.lattice_size,
            token_size=self.token_size,
            memory=self.save_memory,
        )

        ffw = jax.vmap(worker, in_axes=0)(traslational_x).T

        x = ffw * characters

        x = jnp.atleast_1d(x.mean(axis=-1))

        return x


class TraslationAnchor(nn.Module):

    lattice_size: Tuple[int, int]
    TWrap: nn.Module

    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    def setup(self):

        self.wrap = self.Twrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        worker = self.wrap(x)

        # 2D traslational anchoring
        x, anchors = CarreteSign(lattice_size=self.lattice_size, irrep=self.irrep)(x)

        x = jnp.atleast_1d(worker(x))

        # Add phase according to the irrep and the anchor
        x = AddPhase(lattice_size=self.lattice_size, irrep=self.irrep)(x, anchors)

        return x
