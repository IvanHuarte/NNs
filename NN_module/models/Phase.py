import flax.linen as nn
import jax
import jax.typing as jt
import jax.numpy as jnp
import flax
from typing import Tuple
from netket.nn import log_cosh
from .ViT_2D import MultiLayerPerceptron

REAL_DTYPE = jnp.float64

class four_phases_gumbel(nn.Module):
    n_phases: int = 4   # número de fases discretas posibles
    
    @nn.compact
    def __call__(self, x, tau: float = 1.0, key=None):

        phase_logits = self.param("phase_logits", nn.initializers.normal(), (self.n_phases,))
        phase_angles = jnp.array([-0.75, -0.25, 0.25, 0.75]) * jnp.pi

        try:
            key = self.make_rng("phase")
        except flax.errors.InvalidRngError:
            key = None

        # --- Gumbel-softmax para seleccionar fase discreta ---
        if key is None:
            # evaluación determinista: usar argmax
            idx = jnp.argmax(phase_logits)
            global_phase = phase_angles[idx]
        else:
            # entrenamiento: sample diferenciable
            gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, shape=(self.n_phases,))))
            y = nn.softmax((phase_logits + gumbel_noise) / tau)
            global_phase = jnp.sum(y * phase_angles)


        return global_phase
    
class phasors_CNN(nn.Module):

    lattice_size: Tuple

    @nn.compact
    def __call__(self, x):
        # if self.lattice_size[1] == 1:
        #     x.reshape(-1, self.lattice_size[0], x.shape[-1])
        #     kernel =(3,1)
        # else:
        #     x.reshape(-1,*self.lattice_size, x.shape[-1])
        #     kernel =(3,3)

        kernel = (3,3) if self.lattice_size[1] != 1 else (3,1)
        x = x.reshape(-1,*self.lattice_size, 1)

        x = nn.Conv(
            features=64, 
            kernel_size=kernel,
            strides=(1, 1), 
            padding='CIRCULAR',
            #mask=mask,
            dtype=REAL_DTYPE
        )(x)
        x = x.reshape(-1, self.lattice_size[0]*self.lattice_size[1], x.shape[-1])

        # phasors
        # x = nn.glu(                                                                   # Version 1
        # jnp.exp(1j * x).mean(axis=1)
        # ).sum(axis=-1) 

        x = nn.glu(                                                                     # Version 2
            x.mean(axis=1)
        )
        x = jnp.exp(1j*x).sum(axis=-1) 

        return jnp.angle(x)
    
class model_Y(nn.Module):

    lattice_size: Tuple

    @nn.compact
    def __call__(self, x):

        lam_1 = self.param("lambda_1", nn.initializers.normal(), (1,), float)
        lam_2 = self.param("lambda_2", nn.initializers.normal(), (1,), float)
        a = jnp.array([(i + j) % 2 for i in range(self.lattice_size[0]) for j in range(self.lattice_size[1])])
        lam = jnp.where(a == 0, lam_1, lam_2)
        p_real = nn.log_sigmoid(lam * x)
        real_part = 0.5 * jnp.sum(p_real, axis=-1)

        phi_1 = self.param("phi_1", nn.initializers.normal(), (1,), float)
        phi_2 = self.param("phi_2", nn.initializers.normal(), (1,), float)
        phi = jnp.where(a == 0, phi_1, phi_2)
        phase = jnp.pi * nn.sigmoid(phi)
        imag_part = jnp.sum(phase * (x == 1), axis=-1)

        return real_part + 1j * imag_part