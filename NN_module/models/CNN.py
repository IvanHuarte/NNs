import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from typing import Callable, Tuple

REAL_DTYPE = jnp.asarray(1.0).dtype

class TriangularMaskedConv(nn.Module):
    """
    It expects an already padded input
    """
    features: int  
    kernel: Tuple = (3,3)
    strides: Tuple = (1,1)

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        x = x.astype(REAL_DTYPE)
        kernel_shape = (*self.kernel, x.shape[-1], self.features)
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




class ConvBlock(nn.Module):
    """A simple convolutional block with optional batch normalization and pooling.
    It expects an already padded input, so no additional padding is applied.
    The input shape is expected to be (batch_size, height, width, channels).
    """

    features: int
    kernel_size: tuple 
    activation: Callable = nn.tanh
    use_pooling: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Convolución sin padding adicional
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="VALID")(x)
        #x = TriangularMaskedConv(features=self.features)(x)
        
        x = nn.LayerNorm(dtype=REAL_DTYPE)(x)
        x = self.activation(x)
        #print(f"Shape after ConvBlock: {x.shape}")

        if self.use_pooling:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        #print(f"Shape after pooling (if applied): {x.shape}")
        return x


class CNN(nn.Module):
    """A simple 2D Convolutional Neural Network (CNN) model."""
    
    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    block_channels: tuple  # Features for each convolutional block
    kernel_size: tuple     # Size of the convolutional filter

    n_ffn_layers: int = 1  # Number of fully connected layers after convolutional blocks
    activation: Callable = nn.relu  # Activation function
        
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        pad_x = self.kernel_size[0] // 2
        pad_y = self.kernel_size[1] // 2

        x = x.reshape((-1, *self.lattice_size, 1))
        
        if 1 in self.lattice_size:  # If the lattice is 1D, we pad it to make it 2D
            pad_y = 0 

        # Padding periódico manual
        
        #print(f"Input shape after padding: {x.shape}")
        for feature in self.block_channels:
            x = jnp.pad(x, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode="wrap")
            x = ConvBlock(
                features=feature, 
                kernel_size=self.kernel_size,
                use_pooling=True,
                activation=self.activation
                )(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1)) 
        #print(f"Shape after convolutional blocks: {x.shape}")
        
        # Final MLP layers
        for _ in range(self.n_ffn_layers):
            norm = nn.LayerNorm(param_dtype=REAL_DTYPE)
            x = norm(nn.Dense(x.shape[-1])(x))
    
        x = nn.Dense(1, dtype=REAL_DTYPE)(x)  # Output layer

        return x.squeeze()