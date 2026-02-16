import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from ..toolbox import CDense
from NN_module.NN_utils import traslations_2D

DTYPE = jnp.float64


class CMLPWorker(nn.Module):
    """A simple multi-layer perceptron."""

    hidden_alpha: Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, x):

        B = x.shape[0]
        hidden_dims = tuple([int(ha * x.shape[1]) for ha in self.hidden_alpha])

        for hi, act in zip(hidden_dims, self.activation):
            x = CDense(features=hi, param_dtype=DTYPE)(x)
            if callable(act):
                x = act(x)

        if self.final_architecture is not None:
            x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)  # Pooling
            for hi in self.final_architecture:
                x = CDense(features=hi, param_dtype=DTYPE)(x)

        return x


class CMLP_2D(nn.Module):
    """A multi-layer perceptron with 2D-traslational symmetry"""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    activation: Tuple[Callable, ...] = None

    @nn.compact
    def __call__(self, x):
        worker = CMLPWorker(
            hidden_alpha=self.hidden_alpha,
            activation=self.activation,
            final_architecture=self.final_architecture,
        )
        traslational_x = traslations_2D(x, size=self.lattice_size, memory=False)

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class CMLP_Z2(nn.Module):
    """A simple multi-layer perceptron with Z2 symmetry"""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None

    symm_2D: bool = False
    trivial: bool = True

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0] * self.lattice_size[1]
        if self.symm_2D:
            worker = CMLP_2D(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
            )

        else:
            worker = CMLPWorker(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
            )

        output_x = worker(x)
        output_inv_x = worker(-1.0 * x)

        if self.trivial:
            return jax.nn.logsumexp(jnp.array([output_x, output_inv_x]), axis=0)
        else:
            return jax.nn.logsumexp(
                jnp.array([output_x, output_inv_x]), b=jnp.asarray([1.0, -1.0]), axis=0
            )


class CMLP(nn.Module):
    """A batched multi-layer perceptron."""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None

    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):

        N = self.lattice_size[0] * self.lattice_size[1]
        x = (
            x.reshape(x.shape[0], N, x.shape[-1])
            if len(x.shape) >= 3
            else x.reshape(x.shape[0], N)
        )

        if self.symm_Z2:
            worker = CMLP_Z2(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
            )

        elif self.symm_2D:
            worker = CMLP_2D(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
            )

        else:
            worker = CMLPWorker(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
            )

        return worker(x)
