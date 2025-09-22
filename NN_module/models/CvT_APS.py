import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Tuple, Any
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64

def vmap_P_traslations(x, P_shifts):
    
    batched_traslations = lambda x, shift: jnp.roll(x, shift=shift, axis=(0,1))

    return jax.vmap(batched_traslations)(x, P_shifts)

def scan_P_traslations(x, P_shifts):

    def rollit(i, x_batch):
        x_b_trasl = jnp.roll(x_batch, shift=P_shifts[i], axis=(0,1))
        return i+1, x_b_trasl
    
    _, x_trasl = jax.lax.scan(rollit, init=0, xs=x)

    return x_trasl

def polyphase_components(x, strides):

    B, H, W, C = x.shape
    Hd, Wd = H // strides[0], W // strides[1], 
    xt = x.reshape(
        (B ,Hd ,strides[0],Wd,strides[1], C)
        ).transpose((0,1,3,4,2,5)).reshape(
            (B, Hd*Wd, *strides, C)
            ).transpose((0,3,2,1,4))

    shape = xt.shape
    poly_comp = xt.reshape(shape[0], shape[1]*shape[2], shape[3]*shape[4])

    return poly_comp

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
    assert (H % strides[0]==0) & (W % strides[1]==0), f"`lattice_size` must be disible by `strides`. But they are {(H,W)} and {strides}"
    

    poly_comp = polyphase_components(x, strides)

    norm = jnp.linalg.norm(poly_comp, axis=-1)
    flat_idx = jnp.argmax(norm, axis=-1, keepdims=False)
    p, q = jnp.unravel_index(flat_idx, strides) 
    anchors = jnp.array([-p, -q]).T

    return anchors
     

def APS_equivariance_adapter(x, strides, mode = 'vmap'):

    P_shifts = get_maxnorm_indices(x, strides)

    if mode == 'scan':
        x = scan_P_traslations(x, P_shifts)
    elif mode == 'vmap':
        x = vmap_P_traslations(x, P_shifts)
    else:
        raise ValueError(f"No such mode: {mode}")

    return x, P_shifts


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
            x, P_shifts = APS_equivariance_adapter(x, self.strides)

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

        return x, P_shifts
