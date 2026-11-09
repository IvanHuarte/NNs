from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp

RDTYPE = jnp.float64
CDTYPE = jnp.complex128

DTYPE = RDTYPE


class MLP(nn.Module):
    """A simple multi-layer perceptron."""

    hidden_alpha: Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    final_architecture: Tuple[int, ...] | None = None
    only_phase: bool = False

    @nn.compact
    def __call__(self, x):

        B = x.shape[0]
        hidden_dims = tuple(
            [int(ha * x.shape[1] * x.shape[2]) for ha in self.hidden_alpha]
        )

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
                if not jnp.issubdtype(DTYPE, jnp.complexfloating):
                    x = nn.LayerNorm(param_dtype=DTYPE)(x)

        if self.only_phase:
            assert jnp.issubdtype(DTYPE, jnp.complexfloating)
            x = jnp.ones(x.shape, dtype=DTYPE) + 1.0j * x.imag

        return x
