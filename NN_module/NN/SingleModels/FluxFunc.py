import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from typing import Tuple

from netket.nn.activation import log_cosh

from ..toolbox import CDense

DTYPE = jnp.float64


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

        out = jnp.sum(log_cosh(out), axis=-1, keepdims=True)

        return out


class ComplexHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64
    only_phase: bool = False

    @nn.compact
    def __call__(self, x):

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)(
            x.sum(axis=1)
        )

        z = CDense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )(x)

        out = jnp.sum(log_cosh(z), axis=-1, keepdims=True)

        if self.only_phase:
            out = jnp.ones(z.shape, dtype=DTYPE) + 1.0j * z.imag

        return out


class SzaboOutput(nn.Module):

    lattice_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        x_mod = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=DTYPE)(
            x.sum(axis=1)
        )
        log_modulus = nn.Dense(
            features=x.shape[-1],
            param_dtype=DTYPE,
            kernel_init=nn.initializers.lecun_normal(),
        )(x_mod)
        log_modulus = nn.LayerNorm(param_dtype=DTYPE)(log_modulus)

        phase = nn.Conv(
            features=x.shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=DTYPE,
            bias_init=jax.nn.initializers.zeros,
        )(x.reshape(x.shape[0], *self.lattice_size, x.shape[-1]))
        phase = phase.reshape(phase.shape[0], -1, phase.shape[-1])
        # phase = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=DTYPE)(
        #     phase
        # )
        phase = jnp.exp(1j * phase).sum(axis=1)

        nruter = log_modulus + 1j * jnp.angle(phase)
        return jnp.sum(log_cosh(nruter), axis=-1)
