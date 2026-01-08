import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

from NN_module.NN_utils import traslations_2D
from NN_module.models.toolbox import CarreteSign, AddPhase


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

        x = jnp.atleast_2d(x)

        # print(f"Modulus")
        log_modulus = self.Modulus_model(x)

        # print(f"Phase")
        phase = self.Phase_model(x)

        # print(f"Modulus: {log_modulus.shape}")
        # print(f"Phase: {phase.shape}")

        return self.squeeze(log_modulus + 1j * phase)


class SplitTraining_2D(nn.Module):

    ModulusNet: nn.Module
    PhaseNet: nn.Module

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        worker = SplitTraining_Worker(
            ModulusNet=self.ModulusNet, PhaseNet=self.PhaseNet, squeeze=self.squeeze
        )

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

        ffw = jax.vmap(worker, in_axes=0)(traslational_x)

        symm_x = ffw * characters

        return symm_x.mean(axis=0)


class SplitTraining_2DAnchor(nn.Module):

    ModulusNet: nn.Module
    PhaseNet: nn.Module

    lattice_size: Tuple[int, int] = None
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):
        print("x_ini", x.shape)

        worker = SplitTraining_Worker(
            ModulusNet=self.ModulusNet, PhaseNet=self.PhaseNet, squeeze=self.squeeze
        )

        # 2D traslational anchoring
        x, anchors = CarreteSign(lattice_size=self.lattice_size, irrep=self.irrep)(x)
        print("x_anchor", x.shape)

        x = jnp.atleast_1d(worker(x))

        print("x_ffw", x.shape)

        # Add phase according to the irrep and the anchor
        x = AddPhase(lattice_size=self.lattice_size, irrep=self.irrep)(x, anchors)
        print("x_sum", x.shape, "\n")

        return x


class SplitTraining_Z2(nn.Module):

    ModulusNet: nn.Module
    PhaseNet: nn.Module

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
                worker = SplitTraining_2DAnchor(
                    ModulusNet=self.ModulusNet,
                    PhaseNet=self.PhaseNet,
                    lattice_size=self.lattice_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
            else:
                worker = SplitTraining_2D(
                    ModulusNet=self.ModulusNet,
                    PhaseNet=self.PhaseNet,
                    lattice_size=self.lattice_size,
                    token_size=self.token_size,
                    irrep=self.irrep,
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
            worker = SplitTraining_Z2(
                lattice_size=self.lattice_size,
                ModulusNet=self.ModulusNet,
                PhaseNet=self.PhaseNet,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                irrep=self.irrep,
                use_anchor=self.use_anchor,
                token_size=self.token_size,
                squeeze=self.squeeze,
            )
        elif self.symm_2D:
            if self.use_anchor:
                worker = SplitTraining_2DAnchor(
                    ModulusNet=self.ModulusNet,
                    PhaseNet=self.PhaseNet,
                    lattice_size=self.lattice_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
            else:
                worker = SplitTraining_2D(
                    ModulusNet=self.ModulusNet,
                    PhaseNet=self.PhaseNet,
                    lattice_size=self.lattice_size,
                    token_size=self.token_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
        else:
            worker = SplitTraining_Worker(
                ModulusNet=self.ModulusNet, PhaseNet=self.PhaseNet, squeeze=self.squeeze
            )

        x = worker(x)

        return x
