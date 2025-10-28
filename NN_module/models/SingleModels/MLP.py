import netket as nk
import netket.nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any

from NN_module.NN_utils import traslations_2D

DTYPE = jnp.float64


class MultiLayerPerceptron(nn.Module):
    """A simple multi-layer perceptron."""

    param_dtype: Any = DTYPE
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, x):

        B = x.shape[0]

        if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
            normalizer = lambda x: x
        else:
            normalizer = nn.LayerNorm(param_dtype=self.param_dtype)

        hidden_dims = tuple([int(ha * x.shape[0]) for ha in self.hidden_alpha])

        for hi, act in zip(hidden_dims, self.activation):
            x = normalizer(nn.Dense(hi, param_dtype=self.param_dtype)(x))
            if callable(act):
                x = act(x)

        # Works with termination module by default
        if self.final_architecture is None:
            return x

        # To work only with this module, we distinguish between real output
        # and imaginary output (modulus + phase).
        x = x.reshape(B, -1, x.shape[-1])

        if self.two_heads:

            if self.phasors:
                return two_heads_phasors(self.final_architecture)(x)

            else:
                x = x.mean(axis=1)
                x = x.reshape((B, -1))
                return two_heads(self.final_architecture)(x)

        else:

            if self.phasors:
                return glu_phasor()(x)
            else:
                x = x.mean(axis=1)
                x = x.reshape((B, -1))
                return nn.Dense(1)(MultiLayerPerceptron(self.final_architecture)(x))

        return x


class MLP_2D(nn.Module):
    """A multi-layer perceptron with 2D-traslational symmetry"""

    lattice_size: Tuple[int, int]
    param_dtype: Any = DTYPE
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0] * self.lattice_size[1]
        worker = MultiLayerPerceptron(
            N, self.param_dtype, self.hidden_alpha, self.activation, self.output_dim
        )
        traslational_x = traslations_2D(x, size=self.lattice_size, memory=False)

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class MLP_Z2(nn.Module):
    """A simple multi-layer perceptron with Z2 symmetry"""

    lattice_size: Tuple[int, int]
    param_dtype: Any = DTYPE
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    output_dim: int = 1
    trivial: bool = True

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0] * self.lattice_size[1]
        worker = MultiLayerPerceptron(
            N, self.param_dtype, self.hidden_alpha, self.activation, self.output_dim
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
    param_dtype: Any = DTYPE
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1
    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, batched_x):

        if self.symm_Z2:
            worker = MLP_2D(
                self.lattice_size,
                self.param_dtype,
                self.hidden_alpha,
                self.activation,
                self.output_dim,
            )

        elif self.symm_2D:
            worker = MLP_Z2(
                self.lattice_size,
                self.param_dtype,
                self.hidden_alpha,
                self.activation,
                self.output_dim,
                self.trivial_Z2,
            )

        else:
            N = self.lattice_size[0] * self.lattice_size[1]
            worker = MultiLayerPerceptron(
                N, self.param_dtype, self.hidden_alpha, self.activation, self.output_dim
            )

        return jax.vmap(worker, in_axes=0)(batched_x).squeeze()
