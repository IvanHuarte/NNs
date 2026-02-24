import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from typing import Callable, Tuple


class Sum(nn.Module):

    axis: int = -1

    @nn.compact
    def __call__(self, x):
        return jnp.sum(x, axis=self.axis)


class Mean(nn.Module):

    axis: int = -1

    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=self.axis)
