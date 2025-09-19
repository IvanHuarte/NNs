import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple, Any
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64


def polyphase_components(x, strides):

    B, H, W, C = x.shape
    Hd, Wd = (
        H // strides[0],
        W // strides[1],
    )
    # print(f"(B, H, W, C) = {x.shape}")
    # print(f"strides: {strides}")
    # print(f"Hd, Wd = {Hd}, {Wd}")
    x = (
        x.reshape((B, Wd, strides[1], Hd, strides[0], C), order="C")
        .transpose((0, 1, 3, 2, 4, 5))
        .reshape(B, Hd * Wd, *strides, C)
    )
    return x


def get_maxnorm_indices(x, strides):
    """
    This function returns the traslation indices for a batched input of shape (B, H, W, C).
    It computes the polyphase components of a grid for each batch and channel, calculates
    the L2 norm for each component and chooses the indices of the maximum value component.
    It works for both 1D and 2D inputs.
    Input:
        - x: (jnp.ArrayLike) Input data.
        - strides: (tuple) Strides for the next downsampling convolution.

    Returns:
        - x: Shifts (translations) for all batches and channels which makes the input
             traslationaly equivariant.

    """
    _, H, W, _ = x.shape
    assert (H % strides[0] == 0) & (
        W % strides[1] == 0
    ), f"`lattice_size` must be disible by `strides`. But they are {(H,W)} and {strides}"

    poly_comp = polyphase_components(x, strides).transpose((0, 3, 2, 1, 4))
    norm = (
        jnp.linalg.norm(poly_comp, axis=-2, keepdims=True)
        .squeeze(-2)
        .transpose((0, 3, 1, 2))
    )
    norm = norm.reshape(*norm.shape[0:2], norm.shape[2] * norm.shape[3])
    maxnorm_idx = jnp.argmax(norm, axis=-1)[:, :, None]
    row_idx, col_idx = jnp.unravel_index(maxnorm_idx, strides)

    return -jnp.array([row_idx, col_idx]).squeeze(-1).transpose((1, 2, 0))


def APS_equivariance_adapter(x, strides):

    shifts = get_maxnorm_indices(x, strides)
    traslations = lambda x, shift: jnp.roll(x, shift=shift, axis=(0, 1))
    batched_traslations = lambda x, shifts: jax.vmap(traslations)(x, shifts)

    x = jax.vmap(batched_traslations)(x.transpose((0, 3, 1, 2)), shifts)

    return x.transpose((0, 2, 3, 1))


class Conv_APS(nn.Module):

    channels: int
    kernel: Tuple = (3, 3)
    strides: Tuple = (1, 1)
    mask: jt.ArrayLike | None = None
    feature_group_count: int = 1
    padding: str = "SAME"
    dtype: Any = REAL_DTYPE
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        if any([s > 1 for s in self.strides]):  # Downsampling APS correction
            x = APS_equivariance_adapter(x, self.strides)

        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel,
            strides=self.strides,
            mask=self.mask,
            feature_group_count=self.feature_group_count,
            padding=self.padding,
            dtype=self.dtype,
            use_bias=self.use_bias,
        )(x)

        return x
