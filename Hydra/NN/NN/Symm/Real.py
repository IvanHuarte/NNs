import flax.linen as nn
import jax.numpy as jnp


def real_project_logpsi(log_psi):
    """
    log(psi + psi*) up to log 2: log|psi| + log|cos(arg psi)| + i pi [cos(arg psi) < 0].

    Same value as log_psi + log1p(exp(-2i Im log_psi)) (arXiv:2611.18840), written with
    the imaginary part exactly 0 or pi, so it carries no gradient (netket mode="real").
    """
    log_psi = log_psi.astype(jnp.result_type(log_psi.dtype, jnp.complex64))
    c = jnp.cos(log_psi.imag)
    return jnp.log(jnp.abs(c)) + log_psi.real + 1j * jnp.pi * (c < 0)


class Real(nn.Module):
    """
    Time-reversal (real) projection of the wrapped amplitude,

        psi(x) = psi_wrapped(x) + psi_wrapped(x)*,

    exact for a real Hamiltonian in the S^z basis (Heisenberg, J1-J2), whose
    eigenstates can be chosen real. No extra forward pass. The log-derivatives
    are real, so the SR driver can run in mode="real" (half the SRt system).

    Re commutes with every projection of real characters (C4v 1D irreps, Z2,
    momenta k = -k mod the reciprocal lattice: (0,0), (pi,0), (0,pi), (pi,pi)),
    so its place among those wrappers does not matter; with any other momentum
    it mixes k and -k (the generator refuses it). Outermost by convention: the
    final amplitude is real whatever is inside.
    """

    RealWrap: nn.Module

    def setup(self):
        self.worker = self.RealWrap

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return real_project_logpsi(self.worker(x))
