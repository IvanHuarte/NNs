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
    
class DepthPointwiseConv(nn.Module):
    """
    Depthwise pointwise convolution
    """
    channels: int
    kernel: Tuple = (3, 3)
    strides: Tuple = (1, 1)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        
        #Depth-wise convolution
        Ch_in = x.shape[-1]
        x = nn.Conv(
            features=Ch_in,               
            kernel_size=self.kernel,      
            feature_group_count=Ch_in,    
            strides=self.strides,         
            padding='SAME',                   # 'CIRCULAR' para BC periódicas
            use_bias=False
        )
        # Normalizacion
        x = nn.LayerNorm()(x)

        # Point-wise convolution
        x = nn.Conv(
            features=self.channels,   
            kernel_size=(1, 1),           
            strides=(1, 1),                
            padding='SAME',                 # 'CIRCULAR' para BC periódicas
            use_bias=False
        )(x)

        return x
    
class CvTBlock(nn.Module):

    channels: int
    kernel: Tuple = (3,3)
    strides_qkv: Tuple[Tuple, Tuple, Tuple] = ((1,1),(1,1),(1,1))

    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:

        # Convolutional projection
        # x (B,H,W,Ch)
        B, H, W, Ch_in = x.shape
        res1 = x
        x = nn.LayerNorm()(x)
        Q = DepthPointwiseConv(self.channels, kernel=self.kernel, strides=self.strides_qkv[0])(x).reshape(B, -1, self.channels)
        K = DepthPointwiseConv(self.channels, kernel=self.kernel, strides=self.strides_qkv[1])(x).reshape(B, -1, self.channels)
        V = DepthPointwiseConv(self.channels, kernel=self.kernel, strides=self.strides_qkv[2])(x).reshape(B, -1, self.channels)
        

        # Self-attention block
        attn_scores = jnp.matmul(Q,jnp.swapaxes(K,-2,-1)) / jnp.sqrt(self.channels)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_scores = jnp.matmul(attn_weights, V)
        attn_out = attn_out.reshape(B, H, W, self.channels)

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

