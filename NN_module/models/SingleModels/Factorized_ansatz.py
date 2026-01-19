import flax.linen as nn
import jax.numpy as jnp
from typing import Callable

class Factorized(nn.Module):
    """
    General factorized ansatz with optional site-dependent parameters, i.e.

    ψ(x) = ∏ᵢ ψᵢ(xᵢ)

    Therefore,

    log ψ(x) = ∑ᵢ log ψᵢ(xᵢ)

    Module + Phase:

    log |ψ(x)| = ½ ∑ᵢ log[ σ(λᵢ xᵢ) ]

    arg ψ(x) = ∑ᵢ θᵢ · 𝟙_{xᵢ = +1}, with θᵢ = 2π · σ(φᵢ)


    ψ(x) = ∏ᵢ √{ σ(λᵢ xᵢ) } · e^{ i θᵢ 𝟙_{xᵢ = +1} }

    Here σ is the sigmoid function.

    Options:
      - site_dependent=False: λ_i ≡ λ (scalar), φ_i ≡ φ (scalar)
      - site_dependent=True : independent λ_i, φ_i per site
      - complex_wavefunction / positive control whether phase is used.
    """
    lattice_size: tuple
    site_dependent: bool = False
    complex: bool = True
    dtype: jnp.dtype = jnp.float64
    init_kernel: Callable = nn.initializers.lecun_normal()


    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        L = x.shape[-1]

        # Amplitude parameters
        lam_shape = (L,) if self.site_dependent else (1,)
        lam = self.param("lambda", nn.initializers.normal(stddev=1e-2), lam_shape, self.dtype)

        p = nn.log_sigmoid(x * lam)
        real_part = 0.5 * jnp.sum(p, axis=-1)

        # No phase
        if not self.complex:
            return real_part

        # Phase parameters
        phi_shape = (L,) if self.site_dependent else (1,)
        phi = self.param("phi", nn.initializers.normal(stddev=1e-2), phi_shape, self.dtype)
        theta = 2.0 * jnp.pi * nn.sigmoid(phi)

        imag_part = jnp.sum(theta * (x == 1), axis=-1)

        z = real_part.astype(jnp.complex128) + 1j * imag_part.astype(jnp.complex128)

        return jnp.atleast_2d(z).T
