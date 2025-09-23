import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
import flax
from typing import Tuple
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64


class CNNPhasorWorker(nn.Module):

    lattice_size: Tuple
    channels: int

    @nn.compact
    def __call__(self, x):

        kernel = (3, 3) if self.lattice_size[1] != 1 else (3, 1)
        x = x.reshape(-1, *self.lattice_size, 1)

        x = nn.Conv(
            features=self.channels,
            kernel_size=kernel,
            strides=(1, 1),
            padding="CIRCULAR",
            # mask=mask,
            dtype=REAL_DTYPE,
            
        )(x)
        x = x.reshape(-1, self.lattice_size[0] * self.lattice_size[1], x.shape[-1])

        # phasors

        # x = nn.glu(                            # Version 1
        # jnp.exp(1j * x).mean(axis=1)
        # ).sum(axis=-1)

        x = nn.glu(x.mean(axis=1))  # Version 2
        x = jnp.exp(1j * x).sum(axis=-1)

        return jnp.angle(x)

class CNNPhasor_Z2(nn.Module):

    lattice_size: Tuple
    channels: int
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):
        
        worker = CNNPhasorWorker(
                lattice_size=self.lattice_size,
                channels=self.channels
            )

        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Ahora sí podemos concatenar
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            return jax.nn.logsumexp(z2_stack, axis=0, keepdims=False)
        else:
            b = jnp.asarray([1.0, -1.0])[:, None]  # shape (2,1)
            return jax.nn.logsumexp(z2_stack, b=b, axis=0, keepdims=False)



class CNNPhasor(nn.Module):

    lattice_size: Tuple
    channels: int
    symm_Z2: bool = False
    trivial_Z2: bool = True

    @nn.compact
    def __call__(self, x):

        if self.symm_Z2:
            x = CNNPhasor_Z2(
                lattice_size=self.lattice_size,
                channels=self.channels,
                trivial_Z2=self.trivial_Z2
            )(x)

        else:
            x = CNNPhasorWorker(
                lattice_size=self.lattice_size,
                channels=self.channels
            )(x)

        return x