from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from netket.nn.activation import log_cosh

from ..toolbox import CDense

DTYPE = jnp.float64
CDTYPE = jnp.complex128


def sumation(axis):
    def nruter(x):
        return jnp.sum(x, axis=axis)

    return nruter


def product(axis):
    def nruter(x):
        return jnp.prod(x, axis=axis)

    return nruter


def average(axis):
    def nruter(x):
        return jnp.mean(x, axis=axis)

    return nruter


def set_operations(operation, axis):

    if operation == "sum":
        return sumation(axis)
    if operation == "prod":
        return product(axis)
    if operation == "mean":
        return average(axis)


class Sum(nn.Module):

    axis: int = -1
    activation: Callable = None

    @nn.compact
    def __call__(self, x):

        if self.activation is not None:
            x = self.activation[0](x)
        return jnp.sum(x, axis=self.axis)


class Mean(nn.Module):

    axis: int = -1

    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=self.axis)


class FinalFF(nn.Module):

    channels: int
    operation_1: str
    operation_2: str

    def setup(self):
        self.op_1 = set_operations(self.operation_1, axis=(-2, -1))
        self.op_2 = set_operations(self.operation_2, axis=-1)
        self.norm = nn.LayerNorm(param_dtype=DTYPE)

        self.ffw = nn.Dense(
            self.channels, 
            use_bias=True,
            param_dtype=DTYPE,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,    
        )

    def __call__(self, x):
        x = self.op_1(self.norm(x))
        x = self.ffw(x)
        return self.op_2(log_cosh(x)).astype(dtype=CDTYPE)[:, None]


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

        # out = jnp.sum(out, axis=-1, keepdims=True)
        out = jnp.sum(log_cosh(out), axis=-1, keepdims=True)

        return out


class DeepOutputHead(nn.Module):
    d_model: Tuple[int, ...]
    activations_real: Tuple[Callable, ...]
    activations_imag: Tuple[Callable, ...]
    param_dtype = DTYPE
    only_phase: bool = False

    @nn.compact
    def __call__(self, x):

        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x.sum(axis=1))

        x_real = x.copy()
        x_imag = x.copy()

        # Real sequential
        for d, act in zip(self.d_model, self.activations_real):
            norm = nn.LayerNorm(
                use_scale=True, use_bias=True, param_dtype=self.param_dtype
            )
            x_real = nn.Dense(
                d,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )(x_real)
            x_real = norm(x_real)
            if callable(self.activations_real):
                x_real = act(x_imag)

        # Imag sequential
        for d, act in zip(self.d_model, self.activations_imag):
            norm = nn.LayerNorm(
                use_scale=True, use_bias=True, param_dtype=self.param_dtype
            )
            x_imag = nn.Dense(
                d,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )(x_imag)
            x_imag = norm(x_imag)
            if callable(self.activations_real):
                x_imag = act(x_imag)

        z = x_real + 1.0j * x_imag

        if self.only_phase:
            z = jnp.ones(z.shape, dtype=DTYPE) + 1.0j * z.imag

        z = jnp.sum(log_cosh(z), axis=-1, keepdims=True)

        return z


class ComplexHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = DTYPE
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
