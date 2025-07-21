import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
from typing import Callable, Tuple

REAL_DTYPE = jnp.asarray(1.0).dtype

class TriangularMaskedConv(nn.Module):
    """
    It expects an already padded input
    """
    channels: int  
    kernel: Tuple = (3,3)
    strides: Tuple = (1,1)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        x = x.astype(REAL_DTYPE)
        kernel_shape = (*self.kernel, x.shape[-1], self.channels)
        kernel = self.param("kernel", nn.initializers.lecun_normal(), kernel_shape, dtype=REAL_DTYPE)
        mask = jnp.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ])  # Máscara para triangular

        mask = mask[:, :, None, None]  # para broadcast en canales
        masked_kernel = kernel * mask

        return jax.lax.conv_general_dilated(
            x,
            masked_kernel,
            window_strides=self.strides,
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
    
class CvTBlock(nn.Module):

    channels: int
    kernel: Tuple =(3,3)
    strides_qkv: Tuple = ((1,1),(1,1),(1,1))



    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        # Convolutional projection
        # x (B,H,W,Ch)
        res1 = x
        x = nn.LayerNorm()(x)
        Q = TriangularMaskedConv(self.channels, self.strides[0])
        K = TriangularMaskedConv(self.channels, self.strides[1])
        V = TriangularMaskedConv(self.channels, self.strides[2])
        size = (Q.shape[1], Q.shape[2])

        # Self-attention block
        attn_scores = jnp.einsum('bqc,bkc->bqk', Q, K) / jnp.sqrt(self.channels)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bqk,bkc->bqc', attn_weights, V)
        attn_out = attn_out.reshape(-1, *size, self.channels)

        x = res1 + attn_out  # Residual

        # MLP
        res2 = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(int(self.dim * self.mlp_alpha))(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)   

        return res2 + x


    
class StageBlock(nn.Module):
    """
    x = (B, H, W, Ch), where:
        B: Batch size
        H: Height 
        W: Width
        Ch: Number of channels 
    """
    lattice_size = Tuple



    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        # Convolutional token embedding
        x = TriangularMaskedConv(self.CTemb_channels) 

        for _ in range(self.n_blocks):
            x = CvTBlock(dim=self.out_dim, n_heads=self.n_heads)(x)

