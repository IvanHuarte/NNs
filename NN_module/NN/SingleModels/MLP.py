import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from NN_module.NN_utils import traslations_2D

RDTYPE = jnp.float64
CDTYPE = jnp.complex128

DTYPE = CDTYPE


class MLPWorker(nn.Module):
    """A simple multi-layer perceptron."""

    hidden_alpha: Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    only_phase: bool = False

    @nn.compact
    def __call__(self, x):

        B = x.shape[0]
        hidden_dims = tuple([int(ha * x.shape[1]) for ha in self.hidden_alpha])

        for hi, act in zip(hidden_dims, self.activation):
            x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
            if jnp.issubdtype(DTYPE, jnp.floating):
                x = nn.LayerNorm(param_dtype=DTYPE)(x)
            if callable(act):
                x = act(x)

        if self.final_architecture is not None:
            x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)
            for hi in self.final_architecture:
                x = nn.Dense(features=hi, param_dtype=DTYPE)(x)
                if not self.is_complex:
                    x = nn.LayerNorm(param_dtype=DTYPE)(x)

        if self.only_phase:
            assert jnp.issubdtype(DTYPE, jnp.complexfloating)
            x = jnp.ones(x.shape, dtype=DTYPE) + 1.0j * x.imag


        return x


class MLP_2D(nn.Module):
    """A multi-layer perceptron with 2D-traslational symmetry"""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    activation: Tuple[Callable, ...] = None
    only_phase: bool = False


    @nn.compact
    def __call__(self, x):
        worker = MLPWorker(
            hidden_alpha=self.hidden_alpha,
            activation=self.activation,
            final_architecture=self.final_architecture,
            only_phase=self.only_phase,

        )
        traslational_x = traslations_2D(x, size=self.lattice_size, memory=False)

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class MLP_Z2(nn.Module):
    """A simple multi-layer perceptron with Z2 symmetry"""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    only_phase: bool = False


    symm_2D: bool = False
    trivial: bool = True

    @nn.compact
    def __call__(self, x):
        if self.symm_2D:
            worker = MLP_2D(
                lattice_size=self.lattice_size,
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                only_phase=self.only_phase,
            )

        else:
            worker = MLPWorker(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                only_phase=self.only_phase,

            )

        output_x = worker(x)
        output_inv_x = worker(-1.0 * x)

        if self.trivial:
            return jax.nn.logsumexp(jnp.array([output_x, output_inv_x]), axis=0)
        else:
            return jax.nn.logsumexp(
                jnp.array([output_x, output_inv_x]), b=jnp.asarray([1.0, -1.0]), axis=0
            )


class MLP(nn.Module):
    """A batched multi-layer perceptron."""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    only_phase: bool = False


    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):

        x = (
            x.reshape(x.shape[0], x.shape[1]*x.shape[2], *x.shape[3:])
            if len(x.shape) == 4 else x
        )


        if self.symm_Z2:
            worker = MLP_Z2(
                lattice_size=self.lattice_size,
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                only_phase=self.only_phase,
                trivial_Z2=self.trivial_Z2,                
                symm_2D=self.symm_2D,
            )

        elif self.symm_2D:
            worker = MLP_2D(
                lattice_size=self.lattice_size,
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                only_phase=self.only_phase,
            )

        else:
            worker = MLPWorker(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,                
                only_phase=self.only_phase,
            )

        return worker(x)
