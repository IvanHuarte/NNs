import netket as nk
import flax.linen as nn
import jax
import optax
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

def cos_exp_scheduler(epochs, lr0, decay, cycles, n, lr_min):

    exp = lambda step, a: jnp.exp(-a*step/epochs)
    cos = lambda step, cycles: jnp.cos(step*2*jnp.pi/epochs *(cycles-0.5))
    line = lambda step, n: (lr_min-n)/epochs*step+n

    def scheduler_callable(step):
        return (lr0-n)*exp(step,decay)*(1+cos(step,cycles))/2+line(step,n)

    return scheduler_callable

scheduler_dict={
    'cos_exp_scheduler':cos_exp_scheduler,
    'warmup_exponential_decay': optax.warmup_exponential_decay_schedule
}

def scheduler_initializer(name, setup):

    epochs = setup['epochs']
    setup = setup[name]

    if name == 'cos_exp_scheduler':
        return cos_exp_scheduler(
            epochs=epochs,
            lr0=setup['lr0'],
            decay=setup['decay_exp'],
            cycles=setup['cosine_cycles'],
            n=setup['n'],
            lr_min=setup['lr_min']
        )
    elif name == 'warmup_exponential_decay':
        return optax.warmup_exponential_decay_schedule(
            init_value=setup['lr0'],
            peak_value=setup['peak_value'],
            warmup_steps=setup['warmup_steps'],
            transition_steps=1,
            decay_rate=setup['decay_rate'],
        )



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
    size: Tuple[int, int] = (1,1)
) -> npt.ArrayLike:
        """
        Returns a matrix of translations of a flat input vector x 
        which represtents a 2D lattice state. 
        Enhances time execution. 
        """
        x = x.reshape(-1,*size)
        N = size[0] * size[1]
        idx = jnp.unravel_index(jnp.arange(N), size)
        shift_idx = jnp.array(idx).T
        
        roll = lambda x, shift: jnp.roll(x, shift=shift, axis=(-2,-1)).reshape(-1, N)
        
        return jax.vmap(roll,in_axes=(None,0))(x, shift_idx).reshape(-1, N).squeeze()


def traslations_2D_scan(
    x: npt.ArrayLike, 
    size: Tuple[int, int] 
) -> npt.ArrayLike:

    """
    Returns a matrix of translations of a flat input vector x 
    which represtents a 2D lattice state. 
    Enhances memory saving.
    N must be equal to x.shape[-1] 
    """
    N = size[0] * size[1]
    x = x.reshape(size)
    
    def scan_and_roll_x(carry_x, _):
        
        def scan_and_roll_y(carry_y, _):
            y = jnp.roll(carry_y, shift=1, axis=-1)
            return y, y.reshape(-1, N)
        
        _, block_y = jax.lax.scan(scan_and_roll_y, carry_x, length=size[1])
        x = jnp.roll(carry_x, shift=1, axis=-2)

        return x, block_y
    
    return jax.lax.scan(scan_and_roll_x, x, length=size[0])[1].reshape(-1, N).squeeze()



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
    
    if memory:
        x = traslations_2D_scan(x, size)
    else:
        x = traslations_2D_vmap(x, size)

    sub_lat = (size[0] // token_size[0], size[1] // token_size[1])

    # Transforms x into different shapes depending on token_size
    x = x.reshape((-1, sub_lat[1],token_size[1],sub_lat[0],token_size[0]),
                  order='C').transpose((0, 1, 3, 2, 4)).reshape(size[0]*size[1],-1,token_size[0]*token_size[1]).squeeze()

    return x
   


