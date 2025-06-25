
import netket as nk
import netket.nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any

from ..NN_utils import traslations_2D

DTYPE = jnp.complex128

class MultiLayerPerceptron(nn.Module):

    """A simple multi-layer perceptron."""

    N: int = None    
    param_dtype : Any = DTYPE
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1
    #kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        #print("xshapeIN: ", x.shape)
        hidden_dims =tuple([int(ha*self.N) for ha in self.hidden_alpha])

        for hi, act in zip(hidden_dims, self.activation):
            x = nn.Dense(hi,
                         #kernel_init=self.kernel_init,
                         param_dtype=self.param_dtype)(x)
            #x = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            
            if callable(act):
                x = act(x)
        x = nn.Dense(self.output_dim, param_dtype=self.param_dtype)(x)

        return x
    
class MLP_Z2(nn.Module):
    """A simple multi-layer perceptron with Z2 symmetry"""

    lattice_size: Tuple[int, int]  
    param_dtype : Any = DTYPE
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    output_dim: int = 1
    trivial: bool = True   

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0]*self.lattice_size[1]
        worker = MultiLayerPerceptron(
            N,
            self.param_dtype,
            self.hidden_alpha,
            self.activation,
            self.output_dim
        )

        output_x = worker(x)
        output_inv_x = worker(-1.0*x)
        
        if self.trivial:
            return jax.nn.logsumexp(jnp.array([output_x, output_inv_x]), axis = 0)
        else:
            return jax.nn.logsumexp(jnp.array([output_x, output_inv_x]), b=jnp.asarray([1., -1.]), axis = 0)         
            

class MLP_2D(nn.Module):
    """A multi-layer perceptron with 2D-traslational symmetry"""

    lattice_size: Tuple[int, int]      
    param_dtype : Any = DTYPE
    hidden_alpha: Tuple[int, ...] = None
    activation: Tuple[Callable, ...] = None
    output_dim: int = 1        

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0]*self.lattice_size[1]
        worker = MultiLayerPerceptron(
            N,
            self.param_dtype,
            self.hidden_alpha,
            self.activation,
            self.output_dim
        )
        traslational_x = traslations_2D( 
            x,
            size=self.lattice_size,
            memory=False
        )

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)

        
class MLP_2D_Z2(nn.Module):
    """A multi-layer perceptron with both 2D-traslational and Z2 symmetries"""

    lattice_size: Tuple[int, int]    
    param_dtype : Any = jnp.complex64
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1
    trivial: bool = True

    @nn.compact
    def __call__(self, x):
        N = self.lattice_size[0]*self.lattice_size[1]
        worker=MultiLayerPerceptron(
            N,
            self.param_dtype,
            self.hidden_alpha,
            self.activation,
            self.output_dim
        )

        # 2D traslation
        traslational_x = traslations_2D(  # shape = (token_dim, n_tokens, token_dim)
            x,
            size=self.lattice_size,
            memory=False
        )
        
        output_x = jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)
        output_inv_x = jax.vmap(worker, in_axes=0)(-1.0*traslational_x).mean(axis=0)

        if self.trivial:
            return jax.nn.logsumexp(jnp.array([output_x,output_inv_x]), axis=0)
        else:
            return jax.nn.logsumexp(jnp.array([output_x,output_inv_x]), b=jnp.array([1.,-1]), axis=0)


class BatchedMultiLayerPerceptron(nn.Module):

    """A batched multi-layer perceptron."""

    lattice_size: Tuple[int, int]    
    param_dtype : Any = DTYPE
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1
    symm_2D: bool = False
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, batched_x):

        if self.symm_Z2 and self.symm_2D:
            worker = MLP_2D_Z2(
                self.lattice_size,
                self.param_dtype,
                self.hidden_alpha,
                self.activation,
                self.output_dim,
                self.trivial_Z2
            )
            
        elif self.symm_2D and not self.symm_Z2:
            worker = MLP_2D(
                self.lattice_size,
                self.param_dtype,
                self.hidden_alpha,
                self.activation,
                self.output_dim
            )
            
        elif not self.symm_2D and self.symm_Z2:
            worker = MLP_Z2(
                self.lattice_size,
                self.param_dtype,
                self.hidden_alpha,
                self.activation,
                self.output_dim,
                self.trivial_Z2
            )

        else:
            N = self.lattice_size[0]*self.lattice_size[1]
            worker = MultiLayerPerceptron(
                N,
                self.param_dtype,
                self.hidden_alpha,
                self.activation,
                self.output_dim
            )

        return jax.vmap(worker, in_axes=0)(batched_x).squeeze()

