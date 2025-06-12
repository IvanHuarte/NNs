import netket as nk
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy.typing as npt
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "Transformers/transformer_LR_WF_public"))
from transformer_LR_WF.utils import InvertMagnetization


REAL_DTYPE = jnp.asarray(1.0).dtype

activation_dict={
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'softmax': nn.softmax,
    'gelu': nn.gelu,
    'swish': nn.swish,
    'selu': nn.selu,
    'elu': nn.elu,
    'softplus': nn.softplus,
    'relu': nn.relu
}

optimizer_name_dict={
    'Sgd': nk.optimizer.Sgd,
    'adam': nk.optimizer.Adam,
    'AdaGrad': nk.optimizer.AdaGrad
}

sampler_dict={
    'MetropolisLocal': nk.sampler.MetropolisLocal,
    'MetropolisExchange': nk.sampler.MetropolisExchange,
    'MetropolisHamiltonian': nk.sampler.MetropolisHamiltonian,
    'MetropolisSampler': nk.sampler.MetropolisSampler,
}

rule_dict={
    'LocalRule/InvertMagnetization': nk.sampler.rules.MultipleRules([nk.sampler.rules.LocalRule(), InvertMagnetization()],
                                                                    [0.75,0.25]),
}


def circulant(
    row: npt.ArrayLike, times: Optional[int] = None
) -> npt.ArrayLike:
    """Build a (full or partial) circulant matrix based on an array.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        If `times` is None, a square matrix with all the offset versions of the
        first argument. Otherwise, `times` rows of a circulant matrix.
    """
    row = jnp.asarray(row)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1)
        return (new_carry, new_carry)

    if times is None:
        nruter = jax.lax.scan(scan_arg, row, row)[1][::-1, :]
    else:
        nruter = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return nruter


def traslations_2D_vmap(
    x: npt.ArrayLike, 
    token_size: Tuple[int, int] = (1,1)
) -> npt.ArrayLike:
        """
        Returns a matrix of translations of a flat input vector x 
        which represtents a 2D lattice state. 
        Enhances time execution. 
        """

        N = token_size[0] * token_size[1]
        idx = jnp.unravel_index(jnp.arange(N), token_size)
        shift_idx = jnp.array(idx).T
        
        roll = lambda x, shift: jnp.roll(x, shift=shift, axis=(-2,-1)).reshape(-1, N)
        
        return jax.vmap(roll,in_axes=(None,0))(x, shift_idx).reshape(N, -1, N).squeeze()


def traslations_2D_scan(
    x: npt.ArrayLike, 
    token_size: Tuple[int, int] 
) -> npt.ArrayLike:

    """
    Returns a matrix of translations of a flat input vector x 
    which represtents a 2D lattice state. 
    Enhances memory saving.
    N must be equal to x.shape[-1] 
    """
    N = token_size[0] * token_size[1]
    
    def scan_and_roll_x(carry_x, _):
        
        def scan_and_roll_y(carry_y, _):
            y = jnp.roll(carry_y, shift=-1, axis=-1)
            return y, y.reshape(-1, N)
        
        _, block_y = jax.lax.scan(scan_and_roll_y, carry_x, length=token_size[1])
        x = jnp.roll(carry_x, shift=-1, axis=-2)

        return x, block_y
    
    return jax.lax.scan(scan_and_roll_x, x, length=token_size[0])[1].reshape(N, -1, N).squeeze()


def traslations_2D(
    x: npt.ArrayLike,
    size: Tuple[int, int],
    token_size: Tuple[int, int] = None,
    memory: bool = False
) -> npt.ArrayLike:
    
    if token_size is None:
        token_size = size

    if x.shape[0] != size[0] * size[1]:
        raise ValueError("`x` dimension must be equal to `prod(size)`, " +
                         f"but got {x.shape[0]} and {size[0] * size[1]}.")

    sub_lat = (size[0] // token_size[0], size[1] // token_size[1])

    # Transforms x into different shapes depending on token_size
    x = x.reshape((sub_lat[1],token_size[1],sub_lat[0],token_size[0]) ,order='C').transpose((0, 2, 1, 3)).reshape(-1, *token_size)#.squeeze()

    # print(f"Input tokens:\n {x} ") 
    # print(f"Input tokens shape: {x.shape} ")
    # [Creates and] Performs a 2-D tensor of translations for each token 
    if memory:
        return traslations_2D_scan(x, token_size)
    else:
        return traslations_2D_vmap(x, token_size)




  




        


