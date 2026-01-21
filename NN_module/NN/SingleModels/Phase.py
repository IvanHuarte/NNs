import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple, Callable
from NN_module.NN.toolbox import DepthPointwiseConv, MarshallSign

REAL_DTYPE = jnp.float64


class CNNPh(nn.Module):

    lattice_size: Tuple
    channels: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):

        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
        x = x.reshape(-1, *self.lattice_size, 1)

        x = nn.Conv(
            features=self.channels,
            kernel_size=kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            # mask=mask,
            dtype=REAL_DTYPE,
            param_dtype=REAL_DTYPE,
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.lecun_normal(),
        )(x)
        x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1], x.shape[-1])

        x = nn.glu(x.mean(axis=1))  # Version 2
        x = jnp.exp(1j * x).sum(axis=-1)

        return jnp.angle(x)


class EDPPh(nn.Module):
    """
    Embedded Depthwise-Pointwise convolution wit sum over Phasors
    x_inputs must be of shape (B,N,1)

    """

    lattice_size: Tuple
    channels: int
    activation: Callable = nn.swish
    marshall: bool = False

    @nn.compact
    def __call__(self, x_in: jt.ArrayLike) -> jt.ArrayLike:
        assert len(x_in.shape) == 2, f"x.shape: {x_in.shape}"

        N = self.lattice_size[0] * self.lattice_size[1]
        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)

        # Embedding
        x = nn.Embed(
            N,
            self.channels,
            dtype=REAL_DTYPE,
            param_dtype=REAL_DTYPE,
        )(x_in)
        x = x.reshape(-1, *self.lattice_size, self.channels)

        # Depthwise-Normalization-Pointwise
        x = DepthPointwiseConv(self.channels, kernel=kernel)(x)

        x = self.activation(x)
        x = x.reshape(-1, N, x.shape[-1])
        x = nn.Dense(self.channels)(x)

        x = nn.glu(x.mean(axis=1))  # Version JCM
        x = jnp.exp(1j * x).sum(axis=-1)
        phase = jnp.angle(x)[:, None]

        # Marshall sign rule bias
        if self.marshall:
            bias_mars = MarshallSign(lattice_size=self.lattice_size, radians=True)(x_in)
            phase += bias_mars
            phase = (phase + jnp.pi) % (2 * jnp.pi) - jnp.pi

        return phase


class CNNClsf(nn.Module):

    lattice_size: Tuple
    channels: Tuple
    n_classes: int = 2
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x_in):

        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
        x = x_in.reshape(-1, *self.lattice_size, 1)

        for C in self.channels:

            x = nn.Conv(
                features=C,
                kernel_size=kernel,
                strides=(1, 1),
                padding="CIRCULAR",
                # mask=mask,
                dtype=REAL_DTYPE,
                param_dtype=REAL_DTYPE,
                kernel_init=jax.nn.initializers.lecun_normal(),
            )(x)
            x = self.activation(
                nn.LayerNorm(dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(x)
            )

        x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1], x.shape[-1])
        x = x.mean(axis=-1)

        # Clasificador de n clases. Devuelve 0 o pi segun la probabilidad
        x = nn.Dense(self.n_classes)(x)
        x = nn.softmax(x)
        phase = jnp.where(
            x[:, 0] > x[:, 1],
            jnp.array([0.0], dtype=REAL_DTYPE),
            jnp.array([jnp.pi], dtype=REAL_DTYPE),
        )[:, None]
        print(f"phase shape: {phase.shape}")

        return phase


class CNNbinClsf(nn.Module):

    lattice_size: Tuple
    channels: Tuple
    marshall: bool = False
    activation: Callable = nn.swish
    pooling: bool = False

    @nn.compact
    def __call__(self, x_in):

        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
        x = x_in.reshape(-1, *self.lattice_size, 1)

        for C in self.channels:

            x = nn.Conv(
                features=C,
                kernel_size=kernel,
                strides=(1, 1),
                padding="CIRCULAR",
                # mask=mask,
                dtype=REAL_DTYPE,
                param_dtype=REAL_DTYPE,
                kernel_init=jax.nn.initializers.lecun_normal(),
            )(x)
            x = self.activation(
                nn.LayerNorm(dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(x)
            )

        x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1], x.shape[-1])
        x = x.mean(axis=-1)
        # print(f"x shape: {x.shape}")

        # Clasificador binario. 1 salida pasada por sigmoid * pi
        if self.pooling:
            x = jnp.mean(x, axis=-1, keepdims=True)
        else:
            x = nn.Dense(1)(x)
        # print(f"x shape: {x.shape}")

        phase = nn.sigmoid(x) * jnp.pi
        # print(f"phase shape: {phase.shape}")

        # Marshall sign rule bias
        if self.marshall:
            bias_mars = MarshallSign(lattice_size=self.lattice_size, radians=True)(x_in)
            # print(f"bias shape: {bias_mars.shape}")
            phase += bias_mars
            # print(f"phase shape: {phase.shape}")
            phase = (phase + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # print(f"finalx shape: {phase.shape}\n\n")

        return phase


# class CNNSzabo(nn.Module):

#     lattice_size: Tuple
#     channels: int
#     use_bias: bool = False

#     @nn.compact
#     def __call__(self, x):

#         kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
#         x = x.reshape(-1, *self.lattice_size, 1)

#         x = nn.Conv(
#             features=self.channels,
#             kernel_size=kernel,
#             strides=(1, 1),
#             padding="CIRCULAR",
#             # mask=mask,
#             dtype=REAL_DTYPE,
#             param_dtype=REAL_DTYPE,
#             use_bias=self.use_bias,
#             # kernel_init=jax.nn.initializers.lecun_normal(),
#         )(x)
#         x = nn.LayerNorm(dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(x)
#         x = x.reshape(x.shape[0], -1)
#         # x = jnp.exp(1j * jnp.pi * x).sum(axis=-1, keepdims=True)

#         return x.sum(axis=-1, keepdims=True)


class CNNSzabo(nn.Module):

    lattice_size: Tuple
    channels: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):

        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
        x = x.reshape(-1, *self.lattice_size, 1)

        x = nn.Conv(
            features=self.channels,
            kernel_size=kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=REAL_DTYPE,
            param_dtype=REAL_DTYPE,
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.lecun_normal(),
            bias_init=jax.nn.initializers.zeros,
        )(x)
        x = nn.LayerNorm(dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(x)
        x = x.reshape(x.shape[0], -1)
        x = jnp.exp(1j * jnp.pi * x)

        x = x.sum(axis=-1, keepdims=True)

        return jnp.angle(x)


# class CNNSzabo(nn.Module):

#     lattice_size: Tuple
#     channels: int
#     use_bias: bool = False

#     @nn.compact
#     def __call__(self, x):

#         kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
#         x = x.reshape(-1, *self.lattice_size, 1)

#         for _ in range(2):
#             x = nn.Conv(
#                 features=self.channels,
#                 kernel_size=kernel,
#                 strides=(1, 1),
#                 padding="CIRCULAR",
#                 # mask=mask,
#                 dtype=REAL_DTYPE,
#                 param_dtype=REAL_DTYPE,
#                 use_bias=self.use_bias,
#                 kernel_init=jax.nn.initializers.lecun_normal(),
#             )(x)
#             x = nn.LayerNorm(dtype=REAL_DTYPE, param_dtype=REAL_DTYPE)(x)
#             x = nn.relu(x)
#         x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1] * x.shape[-1])
# x = nn.glu(x)
#         x = jnp.exp(-1j * jnp.pi * x).sum(axis=-1, keepdims=True)

#         return jnp.angle(x)
