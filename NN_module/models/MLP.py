
import netket as nk
import netket.nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any

class MultiLayerPerceptron(nn.Module):

    """A simple multi-layer perceptron."""

    N: int = None    
    param_dtype : Any = jnp.complex128
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1


    # def setup(self):
    #     assert self.hidden_alpha is not None, "Hidden_dims must be provided."
    #     assert self.N is not None, "Number of input dimension (N) must be provided."
    #     self.hidden_dims =tuple([int(ha*self.N) for ha in self.hidden_alpha])
    #     if self.activation is None:
    #         self.activation = tuple([0] * len(self.hidden_dims))
    #     else:
    #         if len(self.hidden_dims) != len(self.activation):
    #             raise ValueError("The number of hidden dimensions must match the number of activation functions,"
    #                                 f" but got {len(self.hidden_dims)} and {len(self.activation)} respectively.")

    @nn.compact
    def __call__(self, x):
        #print("xshapeIN: ", x.shape)
        hidden_dims =tuple([int(ha*self.N) for ha in self.hidden_alpha])

        for hi, act in zip(hidden_dims, self.activation):
            x = nn.Dense(hi, param_dtype=self.param_dtype)(x)
            #x = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            if callable(act):
                x = act(x)
        x = nn.Dense(self.output_dim, param_dtype=self.param_dtype)(x)

        #print("xshapeOUT: ", x.shape)

        #return x
        return x

class MultiLayerPerceptron_Traslation(nn.Module):
    """A simple multi-layer perceptron."""

    N: int = None    
    param_dtype : Any = jnp.complex128
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    output_dim: int = 1        

    @nn.compact
    def __call__(self, x):
        MLP = MultiLayerPerceptron(
            self.N,
            self.param_dtype,
            self.hidden_alpha,
            self.activation,
            self.output_dim
        )
        
        batch_size=x.shape[0]
        x_out=jnp.zeros((batch_size, self.output_dim))
        
        for i in range(self.N):
            x_temp = jnp.roll(x, -i)
            x_mlp=MLP(x_temp)
            x_out+=x_mlp

        return x_out
        #return x_out.squeeze(-1)

class MultiLayerPerceptron_Z2_Traslation(nn.Module):
    """A simple multi-layer perceptron."""

    N: int = None    
    param_dtype : Any = jnp.complex64
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        MLP_T=MultiLayerPerceptron_Traslation(
            self.N,
            self.param_dtype,
            self.hidden_alpha,
            self.activation,
            self.output_dim
        )

        y = jnp.array([x , -1.0*x])
        y0 = MLP_T(y[0])
        y1 = MLP_T(y[1])

        return ((y0+y1)/2).squeeze(-1)
        #return (y0+y1)/2
