import netket as nk
import netket.nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any

REAL_DTYPE = jnp.asarray(1.0).dtype


class ConvBlock(nn.Module):
    """A simple convolutional block with optional batch normalization and pooling."""

    features: int
    kernel_size: tuple 
    activation: Callable = nn.tanh
    use_pooling: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Convolución sin padding adicional
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="VALID")(x)
        x = nn.LayerNorm(dtype=REAL_DTYPE)(x)
        x = self.activation(x)

        if self.use_pooling:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
        return x


class CNN(nn.Module):
    """A simple 2D Convolutional Neural Network (CNN) model."""
    
    lattice_size: Tuple[int, int]  # Size of the input image (height, width)

    block_features: tuple  # Features for each convolutional block
    filter_size: tuple     # Size of the convolutional filter

    n_ffn_layers: int = 1  # Number of fully connected layers after convolutional blocks
    activation: Callable = nn.tanh  # Activation function
        
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        pad_x = self.filter_size[0] // 2
        pad_y = self.filter_size[1] // 2

        x = x.reshape((-1, *self.lattice_size, 1))
        
        if 1 in self.lattice_size:  # If the lattice is 1D, we pad it to make it 2D
            pad_y = 0 

        # Padding periódico manual
        x = jnp.pad(x, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode="wrap")
        
        for feature in self.block_features:
            x = ConvBlock(
                features=feature, 
                kernel_size=self.filter_size,
                use_pooling=True,
                activation=nn.tanh
                )(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1)) 
        
        # Final MLP layers
        for _ in range(self.n_ffn_layers):
            norm = nn.LayerNorm(param_dtype=REAL_DTYPE)
            x = self.activation(
                norm(
                    nn.Dense(x.shape[-1])(x)
                    )
            )
        x = nn.Dense(1, dtype=REAL_DTYPE)(x)  # Output layer

        return x.squeeze()