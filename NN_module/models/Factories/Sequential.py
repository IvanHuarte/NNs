import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Any
from frozendict import frozendict as FrozenDict

from NN_module.NN_utils import traslations_2D
from NN_module.models._registry import register_module


class Sequential_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    core_class: nn.Module
    ending_class: nn.Module

    def setup(self):
        self.core = self.core_class
        self.ending = self.ending_class

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        print(f"x_in.shape: {x.shape}")

        x = self.core(x)
        x = self.ending(x)

        return x


class Sequential_2D(nn.Module):

    core_class: nn.Module
    ending_class: nn.Module

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x):

        worker = Sequential_Worker(
            core_class=self.core_class, ending_class=self.ending_class
        )

        # 2D traslation
        print(f"x_before: {x.shape}")
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=self.token_size, memory=False
        )
        print(f"Translational x shape: {traslational_x.shape}")

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class Sequential_Z2(nn.Module):

    core_class: nn.Module
    ending_class: nn.Module

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = Sequential_2D(
                core_class=self.core_class,
                ending_class=self.ending_class,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
            )
        else:
            worker = Sequential_Worker(
                core_class=self.core_class,
                ending_class=self.ending_class,
            )

        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Concatenamos las dos contribuciones
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=0)
            return res
        else:
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)
            return res


@register_module("Sequential")
class Sequential(nn.Module):
    """
    Flax module to have a general model (core) which carries and trains
    global information of the system and ending up with an ending model
    which carries the dimensional reduction and/or modulus-phase spliting.
    """

    core_class: nn.Module
    ending_class: nn.Module

    "Symmetries"
    symm_Z2: bool = False
    symm_2D: bool = False
    trivial_Z2: bool = True

    "Needed for performing 2D traslation symmetries"
    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = Sequential_Z2(
                core_class=self.core_class,
                ending_class=self.ending_class,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
            )
        elif self.symm_2D:
            worker = Sequential_2D(
                core_class=self.core_class,
                ending_class=self.ending_class,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
            )
        else:
            worker = Sequential_Worker(
                core_class=self.core_class,
                ending_class=self.ending_class,
            )

        x = worker(x)

        return x
