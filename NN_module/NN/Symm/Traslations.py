from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from NN_module.NN.toolbox import AddPhase, CarreteSign
from NN_module.utils import Translations2D


class TraslationExplicit(nn.Module):

    TWrap: nn.Module
    lattice_size: Tuple[int, int]

    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    save_memory: bool = False

    def setup(self):

        self.worker = self.TWrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

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

        # 2D translation
        trasl_x = Translations2D(
            x,
            size=self.lattice_size,
            token_size=None,
            memory=self.save_memory,
        )

        trasl_x = trasl_x.reshape(trasl_x.shape[0], x.shape[0], trasl_x.shape[-1])

        ffw = jax.vmap(self.worker, in_axes=0)(trasl_x).transpose((1, 0, 2)).squeeze(-1)
        x = ffw * characters
        x = x.mean(axis=-1, keepdims=True)

        return x


class TraslationAnchor(nn.Module):

    TWrap: nn.Module
    lattice_size: Tuple[int, int]

    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    def setup(self):

        self.worker = self.TWrap
        self.carrete_sign = CarreteSign(
            lattice_size=self.lattice_size, irrep=self.irrep
        )
        self.add_phase = AddPhase(lattice_size=self.lattice_size, irrep=self.irrep)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        # 2D traslational anchoring
        x, anchors = self.carrete_sign(x)

        x = jnp.atleast_1d(self.worker(x))

        # Add phase according to the irrep and the anchor
        x = self.add_phase(x, anchors)

        return x


class Traslation(nn.Module):

    TWrap: nn.Module
    lattice_size: Tuple[int, int]

    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    save_memory: bool = False
    use_anchor: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.use_anchor:
            x = TraslationAnchor(
                lattice_size=self.lattice_size,
                TWrap=self.TWrap,
                irrep=self.irrep,
            )(x)

        else:
            x = TraslationExplicit(
                lattice_size=self.lattice_size,
                TWrap=self.TWrap,
                irrep=self.irrep,
                save_memory=self.save_memory,
            )(x)

        return x
