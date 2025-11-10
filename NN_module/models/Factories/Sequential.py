import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D


class Sequential_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    Seq: Tuple[nn.Module, ...]
    End: nn.Module | None

    squeeze: Callable = lambda x: x

    def setup(self):

        self.seq = self.Seq
        self.end = self.End if (isinstance(self.End, nn.Module)) else lambda x: x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.atleast_2d(x)

        for module in self.seq:
            x = module(x)

        x = self.end(x)

        return self.squeeze(x)


class Sequential_2D(nn.Module):

    Seq: Tuple[nn.Module, ...]
    End: nn.Module
    squeeze: Callable = lambda x: x

    lattice_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x):

        worker = Sequential_Worker(Seq=self.Seq, End=self.End, squeeze=self.squeeze)

        # 2D traslation
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=None, memory=False
        )

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class Sequential_Z2(nn.Module):

    Seq: Tuple[nn.Module, ...]
    End: nn.Module | None
    squeeze: Callable = lambda x: x

    lattice_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = Sequential_2D(
                Seq=self.Seq,
                End=self.End,
                lattice_size=self.lattice_size,
                squeeze=self.squeeze,
            )
        else:
            worker = Sequential_Worker(Seq=self.Seq, End=self.End)

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
    Flax module to have a general model (Seq) which carries and trains
    global information of the system and ending up with an ending model
    which carries the dimensional reduction and/or modulus-phase spliting.
    """

    Seq: Tuple[nn.Module, ...]
    End: nn.Module | None
    squeeze: Callable = lambda x: x

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
                Seq=self.Seq,
                End=self.End,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                lattice_size=self.lattice_size,
                squeeze=self.squeeze,
            )
        elif self.symm_2D:
            worker = Sequential_2D(
                Seq=self.Seq,
                End=self.End,
                lattice_size=self.lattice_size,
                squeeze=self.squeeze,
            )
        else:
            worker = Sequential_Worker(Seq=self.Seq, End=self.End, squeeze=self.squeeze)

        x = worker(x)

        return x
