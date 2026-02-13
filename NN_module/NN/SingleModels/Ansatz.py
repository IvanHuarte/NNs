import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable

from netket.models.jastrow import Jastrow


class Jastrow_wrap(nn.Module):

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(Jastrow()(x)).T

        return x


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
    complex: bool = True
    dtype: jnp.dtype = jnp.float64
    init_kernel: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        L = x.shape[-1]

        # Amplitude parameters
        # lam_shape = (L,) if self.site_dependent else (1,)
        # lam = self.param(
        #     "lambda", nn.initializers.normal(stddev=1e-1), lam_shape, self.dtype
        # )
        # p = nn.log_sigmoid(x * lam)

        # Phase parameters
        # phi_params = self.param(
        #     "phi", nn.initializers.normal(stddev=1e-1), (2,), self.dtype
        # )
        # i, j = jnp.unravel_index(jnp.arange(L), self.lattice_size)
        # phi = phi_params[(i + j) % 2]
        # theta = jnp.pi * nn.sigmoid(phi)

        # phi_params = self.param(
        #     "phi", nn.initializers.normal(stddev=1e-1), (2,), self.dtype
        # )
        # phi_shape = (L,) if self.site_dependent else (1,)
        # imag_part = jnp.sum(theta * (x == 1), axis=-1)

        phi_shape = (L,)
        phi = self.param("phi", nn.initializers.normal(stddev=5e-2), phi_shape, self.dtype)
        theta = jnp.pi * nn.sigmoid(phi)
        imag_part = jnp.sum(theta * (x == 1), axis=-1)

        # Add constant modulus
        if self.complex:
            real_part = jnp.array([0.5])
            z = real_part.astype(jnp.complex128) + 1j * imag_part.astype(jnp.complex128)
            return z
        else:
            return jnp.atleast_2d(imag_part).T


class FactorMod(nn.Module):
    """
    General factorized ansatz modified.

    """

    lattice_size: tuple
    dtype: jnp.dtype = jnp.float64
    init_kernel: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        L = x.shape[-1]

        # g(x) gating for a NN module
        x = nn.Dense(
            L,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(x)

        # x = nn.glu(x)
        x = jnp.sum(x, axis=-1)
        x = jnp.tanh(x)

        x = jnp.atleast_2d(x).T

        return x
