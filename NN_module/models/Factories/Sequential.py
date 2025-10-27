import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Any
from frozendict import frozendict as FrozenDict

from NN_module.NN_utils import traslations_2D

class Sequential_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    Core: nn.Module
    End: nn.Module | None

    def setup(self):

        self.core = self.Core
        self.end = self.End if (
            isinstance(self.End, nn.Module)
            ) else lambda x: x 

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = self.core(x)
        x = self.end(x)

        return x


class Sequential_2D(nn.Module):

    Core: nn.Module
    End: nn.Module

    lattice_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x):

        worker = Sequential_Worker(
            Core=self.Core, 
            End=self.End
        )

        # 2D traslation
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=None, memory=False
        )

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class Sequential_Z2(nn.Module):

    Core: nn.Module
    End: nn.Module | None

    lattice_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = Sequential_2D(
                Core=self.Core, 
                End=self.End,
                lattice_size=self.lattice_size,
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


class Sequential(nn.Module):
    """
    Flax module to have a general model (core) which carries and trains
    global information of the system and ending up with an ending model
    which carries the dimensional reduction and/or modulus-phase spliting.
    """

    Core: nn.Module
    End: nn.Module | None

    "Symmetries"
    symm_Z2: bool = False
    symm_2D: bool = False
    trivial_Z2: bool = True

    "Needed for performing 2D traslation symmetries"
    lattice_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = Sequential_Z2(
                Core=self.Core, 
                End=self.End,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                lattice_size=self.lattice_size
            )
        elif self.symm_2D:
            worker = Sequential_2D(
                Core=self.Core, 
                End=self.End,
                lattice_size=self.lattice_size
            )
        else:
            worker = Sequential_Worker(
                Core=self.Core, 
                End=self.End
            )

        x = worker(x)

        return x
