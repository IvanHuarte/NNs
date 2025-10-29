import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D


class SplitTraining_Worker(nn.Module):
    """
    Flax module to train modulus and phase separately
    """

    ModulusNet: nn.Module
    PhaseNet: nn.Module

    squeeze: Callable = lambda x: x

    def setup(self):
        self.Modulus_model = self.ModulusNet
        self.Phase_model = self.PhaseNet

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        log_modulus = self.Modulus_model(x)
        phase = self.Phase_model(x)
        # print(log_modulus.shape)
        # print(phase.shape)

        phi = log_modulus + 1j * phase

        return self.squeeze(phi)


class SplitTraining_2D(nn.Module):

    ModulusNet: nn.Module
    PhaseNet: nn.Module

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        worker = SplitTraining_Worker(
            ModulusNet=self.ModulusNet, PhaseNet=self.PhaseNet, squeeze=self.squeeze
        )

        # 2D traslation
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=self.token_size, memory=False
        )

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class SplitTraining_Z2(nn.Module):

    ModulusNet: nn.Module
    PhaseNet: nn.Module

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    trivial_Z2: bool = False

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = SplitTraining_2D(
                ModulusNet=self.ModulusNet,
                PhaseNet=self.PhaseNet,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                squeeze=self.squeeze,
            )
        else:
            worker = SplitTraining_Worker(
                ModulusNet=self.ModulusNet, PhaseNet=self.PhaseNet, squeeze=self.squeeze
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


class SplitTraining(nn.Module):
    """
    Flax module to train modulus and phase separately
    """

    ModulusNet: nn.Module
    PhaseNet: nn.Module

    "Symmetries"
    symm_Z2: bool = False
    symm_2D: bool = False
    trivial_Z2: bool = True

    "Needed for performing 2D traslation symmetries"
    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = SplitTraining_Z2(
                ModulusNet=self.ModulusNet,
                PhaseNet=self.PhaseNet,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                squeeze=self.squeeze,
            )
        elif self.symm_2D:
            worker = SplitTraining_2D(
                ModulusNet=self.ModulusNet,
                PhaseNet=self.PhaseNet,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
                squeeze=self.squeeze,
            )
        else:
            worker = SplitTraining_Worker(
                ModulusNet=self.ModulusNet, PhaseNet=self.PhaseNet, squeeze=self.squeeze
            )

        x = worker(x)

        return x
