import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from typing import Callable, Tuple

from netket.nn.activation import log_cosh

from NN_module.NN.SingleModels.MLP import DTYPE


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


class OutputHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64
    only_phase: bool = False

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j * out_imag

        if self.only_phase:
            out = jnp.ones(out.shape, dtype=DTYPE) + 1.0j * out.imag

        return jnp.sum(log_cosh(out), axis=-1)
