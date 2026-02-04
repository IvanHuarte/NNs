import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D
from NN_module.NN.toolbox import CarreteSign, AddPhase


class Sequential_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    Seq: Tuple[nn.Module, ...]
    ZZ: nn.Module | None

    squeeze: Callable = lambda x: x

    def setup(self):

        self.seq = self.Seq
        self.end = self.ZZ if (isinstance(self.ZZ, nn.Module)) else lambda x: x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.atleast_2d(x)

        for module in self.seq:
            x = module(x)

        x = self.end(x)

        return self.squeeze(x)


class Sequential_2D(nn.Module):

    Seq: Tuple[nn.Module, ...]
    ZZ: nn.Module

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        worker = Sequential_Worker(Seq=self.Seq, ZZ=self.ZZ, squeeze=self.squeeze)

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
            x, size=self.lattice_size, token_size=self.token_size, memory=False
        )

        ffw = jax.vmap(worker, in_axes=0)(traslational_x).T

        x = ffw * characters

        x = jnp.atleast_1d(x.mean(axis=-1))

        return x


class Sequential_2DAnchor(nn.Module):

    Seq: Tuple[nn.Module, ...]
    ZZ: nn.Module

    lattice_size: Tuple[int, int] = None
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        worker = Sequential_Worker(Seq=self.Seq, ZZ=self.ZZ, squeeze=self.squeeze)

        # 2D traslational anchoring
        x, anchors = CarreteSign(lattice_size=self.lattice_size, irrep=self.irrep)(x)

        x = jnp.atleast_1d(worker(x))

        # Add phase according to the irrep and the anchor
        x = AddPhase(lattice_size=self.lattice_size, irrep=self.irrep)(x, anchors)

        return x


class Sequential_Z2(nn.Module):

    Seq: Tuple[nn.Module, ...]
    ZZ: nn.Module | None

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    use_anchor: bool = False

    trivial_Z2: bool = False

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            if self.use_anchor:
                worker = Sequential_2DAnchor(
                    Seq=self.Seq,
                    ZZ=self.ZZ,
                    lattice_size=self.lattice_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )

            else:
                worker = Sequential_2D(
                    Seq=self.Seq,
                    ZZ=self.ZZ,
                    lattice_size=self.lattice_size,
                    token_size=self.token_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
        else:
            worker = Sequential_Worker(Seq=self.Seq, ZZ=self.ZZ)

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
    ZZ: nn.Module | None

    "Symmetries"
    symm_2D: bool = False
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    use_anchor: bool = False

    symm_Z2: bool = False
    trivial_Z2: bool = True

    "Needed for performing 2D traslation symmetries"
    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = Sequential_Z2(
                Seq=self.Seq,
                ZZ=self.ZZ,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                irrep=self.irrep,
                use_anchor=self.use_anchor,
                token_size=self.token_size,
                squeeze=self.squeeze,
            )
        elif self.symm_2D:
            if self.use_anchor:
                worker = Sequential_2DAnchor(
                    Seq=self.Seq,
                    ZZ=self.ZZ,
                    lattice_size=self.lattice_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )

            else:
                worker = Sequential_2D(
                    Seq=self.Seq,
                    ZZ=self.ZZ,
                    lattice_size=self.lattice_size,
                    token_size=self.token_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
        else:
            worker = Sequential_Worker(Seq=self.Seq, ZZ=self.ZZ, squeeze=self.squeeze)

        x = jnp.atleast_1d(worker(x))

        return x
