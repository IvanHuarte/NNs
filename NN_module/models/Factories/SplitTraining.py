import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Any
from frozendict import frozendict as FrozenDict

from NN_module.NN_utils import traslations_2D
from NN_module.models._registry import register_module


class SplitTraining_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    modulus_clss: nn.Module
    phase_clss: nn.Module

    # modulus_setup: FrozenDict[str, Any]  # Frozen and hashable dict
    # phase_setup: FrozenDict[str, Any]  # Frozen and hashable dict

    def setup(self):
        self.mod_model = self.modulus_clss(
            name="modulus",
            # **self.modulus_setup
        )
        self.ph_model = self.phase_clss(
            name="phase"
            # **self.phase_setup
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        print(f"x_in.shape: {x.shape}")

        log_modulus = self.mod_model(x)
        phase = self.ph_model(x)

        return log_modulus + 1j * phase


class SplitTraining_2D(nn.Module):

    modulus_clss: nn.Module
    phase_clss: nn.Module

    # modulus_setup: FrozenDict[str, Any]  # Frozen and hashable dict
    # phase_setup: FrozenDict[str, Any]  # Frozen and hashable dict

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x):

        worker = SplitTraining_Worker(
            modulus_clss=self.modulus_clss,
            phase_clss=self.phase_clss,
            # modulus_setup=self.modulus_setup,
            # phase_setup=self.phase_setup,
        )

        # 2D traslation
        print(f"x_before: {x.shape}")
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=self.token_size, memory=False
        )
        print(f"Translational x shape: {traslational_x.shape}")

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class SplitTraining_Z2(nn.Module):

    modulus_clss: nn.Module
    phase_clss: nn.Module

    # modulus_setup: FrozenDict[str, Any]  # Frozen and hashable dict
    # phase_setup: FrozenDict[str, Any]  # Frozen and hashable dict

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = SplitTraining_2D(
                modulus_clss=self.modulus_clss,
                phase_clss=self.phase_clss,
                # modulus_setup=self.modulus_setup,
                # phase_setup=self.phase_setup,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
            )
        else:
            worker = SplitTraining_Worker(
                modulus_clss=self.modulus_clss,
                phase_clss=self.phase_clss,
                # modulus_setup=self.modulus_setup,
                # phase_setup=self.phase_setup,
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


@register_module("SplitTraining")
class SplitTraining(nn.Module):
    """
    Flax module to train module and phase separately
    """

    modulus_clss: nn.Module
    phase_clss: nn.Module

    # modulus_setup: FrozenDict[str, Any]  # Frozen and hashable dict
    # phase_setup: FrozenDict[str, Any]  # Frozen and hashable dict

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
            worker = SplitTraining_Z2(
                modulus_clss=self.modulus_clss,
                phase_clss=self.phase_clss,
                # modulus_setup=self.modulus_setup,
                # phase_setup=self.phase_setup,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
            )
        elif self.symm_2D:
            worker = SplitTraining_2D(
                modulus_clss=self.modulus_clss,
                phase_clss=self.phase_clss,
                # modulus_setup=self.modulus_setup,
                # phase_setup=self.phase_setup,
                lattice_size=self.lattice_size,
                token_size=self.token_size,
            )
        else:
            worker = SplitTraining_Worker(
                modulus_clss=self.modulus_clss,
                phase_clss=self.phase_clss,
                # modulus_setup=self.modulus_setup,
                # phase_setup=self.phase_setup,
            )

        x = worker(x)

        return x
