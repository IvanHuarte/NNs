from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PromoteDtypeFn
from flax.typing import (
    Dtype,
    Initializer,
)
from flax.typing import (
    PRNGKey as PRNGKey,
)
from flax.typing import (
    Shape as Shape,
)

from NN_module.utils import Translations2D

REAL_DTYPE = jnp.float64


def get_mask(name) -> jnp.ndarray:

    if name is None:
        return None

    if name == "Triangular":
        return jnp.array(
            [  # Mascara para red triangular
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ]
        )
    if name == "Square":
        return jnp.array(
            [  # Mascara para red cuadrada
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )
    else:
        return jnp.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )


def get_min_idx(x):
    """
    Input: x=x(s_i) (s_i = +1,-1) of shape (B, N)  where B
    is the set of all traslation irreps and N is the
    configuration dimension. Returns the anchor index as the
    minimum value calculated as the integer representation
    of each configuration in its bit form.
    """
    bin_x = (-(x - 1) / 2).astype(jnp.int8)
    bin_x = bin_x.T[::-1].T
    powers = jnp.tile(jnp.arange(x.shape[-1]), (x.shape[0], 1))
    idx = jnp.sum(bin_x * 2**powers, axis=-1)
    min_idx = jnp.argmin(idx)

    return min_idx


def batched_get_anchor(size, memory=True):
    def core(_, x):
        x = jnp.atleast_2d(x)  # (1, N)
        x = Translations2D(x, size, memory=memory)  # (N, N)
        idx = get_min_idx(x)
        return _, idx

    return core


class CDense(nn.Module):
    features: int
    use_bias: bool = True
    param_dtype: Dtype = jnp.float64
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros_init()
    promote_dtype: PromoteDtypeFn = promote_dtype

    @nn.compact
    def __call__(self, x):
        cdtype = jnp.complex128 if self.param_dtype == jnp.float64 else jnp.complex64

        kernel = self.param(
            "ckernel",
            self.kernel_init,
            (x.shape[-1], self.features, 2),
            self.param_dtype,
        )
        ckernel = (kernel[:, :, 0] + 1j * kernel[:, :, 1]).astype(dtype=cdtype)

        if self.use_bias:
            bias = self.param(
                "cbias",
                self.bias_init,
                (self.features, 2),
                self.param_dtype,
            )
            cbias = (bias[:, 0] + 1j * bias[:, 1]).astype(dtype=cdtype)
        else:
            cbias = None

        x, ckernel, cbias = self.promote_dtype(x, ckernel, cbias, dtype=None)

        assert x is not None
        assert kernel is not None

        y = jax.lax.dot_general(
            x,
            ckernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=None,
        )
        if cbias is not None:
            y += jnp.reshape(cbias, (1,) * (y.ndim - 1) + (-1,))

        return y


class MultiLayerPerceptron(nn.Module):
    """
    Flax module for a Multi-layer perceptron architecture with normalization.

    Args:
        layer_widths: Sequence of integers that define both the number of layers
            and their widths.
        activation_funcion: The activation function that will be applied after
            each dense layer.
        kernel_init: Function to initialize the trainable parameters.

    Returns:
        A jax array with the output of the net.
    """

    layer_widths: Sequence[int]
    activation_function: Callable = nn.swish
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        for w in self.layer_widths:
            # We cannot use LayerNorm when the output has size 1, since
            # that would destroy the data.
            if w == 1:
                normalizer = lambda x: x
            else:
                normalizer = nn.LayerNorm(param_dtype=REAL_DTYPE)
            x = self.activation_function(
                normalizer(
                    nn.Dense(
                        w,
                        kernel_init=self.kernel_init,
                        param_dtype=REAL_DTYPE,
                    )(x)
                )
            )
        return x


class DepthPointwiseConv(nn.Module):
    """
    Depthwise pointwise convolution
    """

    channels: int
    kernel: tuple = (3, 3)
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        Ch_in = x.shape[-1]

        # Depth-wise convolution (Aplica mascara adyacente a cada canal)
        x = nn.Conv(
            features=Ch_in,
            kernel_size=self.kernel,
            feature_group_count=Ch_in,
            strides=(1, 1),
            padding="CIRCULAR",
            dtype=REAL_DTYPE,
            use_bias=False,
        )(x)
        # Normalizacion
        x = nn.LayerNorm()(x)

        # Point-wise convolution
        x = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=REAL_DTYPE,
            use_bias=False,
        )(x)

        return x


class MarshallSign(nn.Module):
    """Marshall sign for 2D square lattices

    Args:
        x: input array of shape (B,N)
    Returns:
        array of shape (B,) with the Marshall sign

    """

    lattice_size: tuple
    radians: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        x = x.reshape(-1, *self.lattice_size)

        Lx, Ly = self.lattice_size
        xs = jnp.arange(Lx).reshape(-1, 1) + jnp.arange(Ly).reshape(1, -1)
        marshall_pattern = (-1) ** (xs % 2)
        sign = jnp.prod(jnp.where(x > 0, marshall_pattern, 1), axis=(1, 2))

        if self.radians:
            sign = jnp.where(sign < 0, jnp.pi, 0.0)

        sign = sign.reshape(-1, 1)

        return sign


class CarreteSign(nn.Module):

    lattice_size: tuple
    save_mem: bool = True
    irrep: tuple[int] = (0, 0)

    def setup(self):

        self.size = self.lattice_size
        self.N = self.size[0] * self.size[1]
        self.shifts = jnp.array(
            jnp.unravel_index(jnp.arange(self.N), self.lattice_size)
        ).T
        self.mem = self.save_mem

    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        B, N = x.shape

        if self.save_mem:
            idx = jax.lax.scan(
                batched_get_anchor(self.size), init=0, xs=x, length=x.shape[0]
            )[1]

        else:
            x = (
                Translations2D(x, self.size, memory=True)
                .reshape(N, B, N)
                .transpose((1, 0, 2))
            )
            idx = jax.vmap(get_min_idx, in_axes=0)(x)

        anchor = self.shifts[idx]

        return x, anchor


class AddPhase(nn.Module):
    """Adds a phase according to a given irrep for 2D lattices.

    Args:
        x: input array of shape (B,)
        anchors: array of shape (B, 2) with the anchor points
        irrep: Tuple (q_1, q_2) representing the irrep.
    Returns:
        array of shape (B,) with the added phase
    """

    lattice_size: tuple[int, int]
    irrep: tuple[int, int]

    @nn.compact
    def __call__(self, x: jt.ArrayLike, anchors: jt.ArrayLike) -> jt.ArrayLike:

        phase = (
            2
            * jnp.pi
            * (
                self.irrep[0] * anchors[:, 0] / self.lattice_size[0]
                + self.irrep[1] * anchors[:, 1] / self.lattice_size[1]
            )
        )

        new_phase = (x.imag + phase + jnp.pi) % (2 * jnp.pi) - jnp.pi

        return (x.real + 1j * new_phase).astype(x.dtype)
