import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any

from ..toolbox import CDense
from NN_module.NN_utils import traslations_2D

RDTYPE = jnp.float64
CDTYPE = jnp.complex128

class MLPWorker(nn.Module):
    """A simple multi-layer perceptron."""

    hidden_alpha: Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    dense_backend: str = "real"

    @nn.compact
    def __call__(self, x):
        print(f"Hidd: {self.hidden_alpha}")
        print(f"Act: {self.activation}")
        print(f"Final: {self.final_architecture}")
        print(f"Dense_back: {self.dense_backend}")

        def make_dense(features):
            if self.dense_backend == "real":
                return nn.Dense(features=features, param_dtype=RDTYPE)
            elif self.dense_backend == "complex":
                return nn.Dense(features=features, param_dtype=CDTYPE)
            elif self.dense_backend == "cdense":
                return CDense(features=features, param_dtype=RDTYPE)

        if self.dense_backend == "real":
            is_complex = False
            param_dtype = RDTYPE
        elif self.dense_backend == "complex":
            is_complex = True
            param_dtype = CDTYPE
        elif self.dense_backend == "cdense":
            is_complex = True
            param_dtype = RDTYPE
        else:
            raise ValueError(f"Unknown dense_backend: {self.dense_backend}")

        B = x.shape[0]
        hidden_dims = tuple([int(ha * x.shape[1]) for ha in self.hidden_alpha])

        for hi, act in zip(hidden_dims, self.activation):
            x = make_dense(hi)(x)
            if not is_complex:
                x = nn.LayerNorm(param_dtype=param_dtype)(x)
            if callable(act):
                x = act(x)

        if self.final_architecture is not None:
            # Pooling si es necesario
            x = x.reshape(B, -1, x.shape[-1]).mean(axis=1)
            for hi in self.final_architecture:
                x = make_dense(hi)(x)
                if not is_complex:
                    x = nn.LayerNorm(param_dtype=param_dtype)(x)

        return x


class MLP_2D(nn.Module):
    """A multi-layer perceptron with 2D-traslational symmetry"""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    activation: Tuple[Callable, ...] = None
    dense_backend: str = "real"

    @nn.compact
    def __call__(self, x):
        worker = MLPWorker(
            hidden_alpha=self.hidden_alpha,
            activation=self.activation,
            final_architecture=self.final_architecture,
            dense_backend=self.dense_backend
        )
        traslational_x = traslations_2D(x, size=self.lattice_size, memory=False)

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class MLP_Z2(nn.Module):
    """A simple multi-layer perceptron with Z2 symmetry"""

    lattice_size: Tuple[int, int]
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    dense_backend: str = "real"

    symm_2D: bool = False
    trivial: bool = True

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0] * self.lattice_size[1]
        if self.symm_2D:
            worker = MLP_2D(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                dense_backend=self.dense_backend

            )

        else:
            worker = MLPWorker(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                dense_backend=self.dense_backend
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
    dense_backend: str = "real"

    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):

        N = self.lattice_size[0] * self.lattice_size[1]
        x = x.reshape(x.shape[0], N)

        if self.symm_Z2:
            worker = MLP_Z2(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                dense_backend=self.dense_backend,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
            )

        elif self.symm_2D:
            worker = MLP_2D(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                dense_backend=self.dense_backend
            )

        else:
            worker = MLPWorker(
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                final_architecture=self.final_architecture,
                dense_backend=self.dense_backend
            )

        return worker(x)
