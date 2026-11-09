import flax.linen as nn
import jax.numpy as jnp
from netket.jax import logsumexp_cplx

from Hydra.utils import _angle


def get_characters(
    lattice_size: tuple[int, int], irrep: tuple[int, int]
) -> jnp.ndarray:
    """
    Get the characters of the irrep for the given input x
    """
    xs = jnp.arange(0, lattice_size[1])
    ys = jnp.arange(0, lattice_size[0])

    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    indexes = jnp.stack([yy, xx], axis=-1)

    n = indexes[..., 0].flatten()
    m = indexes[..., 1].flatten()

    characters = jnp.exp(
        2.0j
        * jnp.pi
        * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    ).reshape(1, *lattice_size)

    # characters = 2.0j * jnp.pi * (irrep[0] * n / lattice_size[0] + irrep[1] * m / lattice_size[1])
    # characters = characters.reshape(1, *lattice_size)
    # print(characters)

    return characters


PHASE_MODES = ("angle", "log")


class Sequivariant(nn.Module):
    """
    Flax module to project an equivariant module to an irrep.

    z = sum_r chi_k(r) f_0(r) over the backbone output grid (channel 0).

    phase_mode:
        "angle" (legacy)  log psi = head + i arg(z). The amplitude does not
                          vanish where z -> 0 while the phase winds there:
                          psi is discontinuous and O_k ~ 1/|z| is not damped
                          by the sampling weight.
        "log"             log psi = head + log(z), i.e. psi = exp(head) * z.
                          psi vanishes continuously at z = 0 and |psi|^2
                          cancels the 1/|z|^2 of the log-derivatives.
                          Same parameters as "angle" (checkpoints compatible).

    coset: (sy, sx) or None
        A backbone with stride/tokens (sy, sx) is only equivariant under
        translations by (sy, sx): the projection above fixes the phase of
        T_(sy,sx) only, mixing momenta k + 2 pi (a/sy, b/sx). With coset the
        output is summed over the sy*sx translations inside a token,
            psi_k(x) = sum_t exp(-i k.t) psi(T_t x),
        which is an exact momentum eigenstate of the full lattice. irrep is
        then the momentum of the full lattice (n_y, n_x), k = 2 pi n / L.
        Costs sy*sx forward passes (done as one batch).
    """

    SeqV: tuple[nn.Module, ...]
    irrep: tuple[int, int]
    eps: float = 1e-10
    z_eps: float = 1e-12 + 1e-15j
    phase_mode: str = "angle"
    coset: tuple[int, int] | None = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.coset is None or tuple(self.coset) == (1, 1):
            return self._project(x)

        sy, sx = self.coset
        B, Ly, Lx = x.shape[:3]
        shifts = [(ty, tx) for ty in range(sy) for tx in range(sx)]
        xt = jnp.concatenate([jnp.roll(x, t, axis=(1, 2)) for t in shifts], axis=0)
        log_psi = self._project(xt).reshape(len(shifts), B, -1)  # (T, B, 1)

        phases = jnp.array(
            [2 * jnp.pi * (self.irrep[0] * ty / Ly + self.irrep[1] * tx / Lx)
             for ty, tx in shifts]
        )[:, None, None]
        return logsumexp_cplx(log_psi - 1j * phases, axis=0)

    def _project(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode={self.phase_mode!r}, expected one of {PHASE_MODES}")

        for module in self.SeqV[:-1]:
            x = module(x)

        _, H, W, _ = x.shape

        characters = get_characters((H, W), self.irrep)  # (B, H, W)

        # Get Theta_K
        z = (characters * x[..., 0].astype(jnp.complex128)).sum(axis=(1, 2))

        log_psi = self.SeqV[-1](x)  # log_psi = (B, 1)
        if self.phase_mode == "angle":
            phase_k = _angle(z + self.z_eps)[:, None]  # phase_k = (B, 1)
            log_psi_k = log_psi + 1j * phase_k
        else:
            z_safe = jnp.where(jnp.abs(z) < self.eps, 1.0 + 0j, z)
            log_psi_k = log_psi + jnp.log(z_safe)[:, None]

        min_infinity = jnp.array(
            [[jnp.finfo(log_psi.dtype).min + 0j]], dtype=jnp.complex128
        )
        min_infinity = jnp.broadcast_to(min_infinity, log_psi_k.shape)

        log_psi_k = jnp.where(
            (jnp.abs(z) < self.eps)[:, None],
            min_infinity,
            log_psi_k,
        )

        # print(f"{x[..., 0]=}")
        # print(f"{characters=}")
        # print(f"char * x = \n{characters * x[..., 0]}")
        # print(f"{z=}")
        # print(f"{log_psi=}")
        # print(f"{phase_k=}")
        # print(f"{log_psi_k=}")
        # print("\n")

        return log_psi_k


class SequivariantX(nn.Module):
    """
    Flax module to project an equivariant module to an irrep
    """

    SeqVX: tuple[nn.Module, ...]
    irrep: tuple[int, int]
    eps: float = 1e-6

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        _, H, W, _ = x.shape

        characters = get_characters((H, W), self.irrep)  # (B, H, W)

        # Get Theta_K
        z = (characters * x[..., 0].astype(jnp.complex128)).sum(axis=(1, 2))
        z_safe_point = jnp.broadcast_to(jnp.array([1.0 +1.0j], dtype=z.dtype), z.shape)
        z_safe = jnp.where(jnp.abs(z) < self.eps, z_safe_point, z)
        phase_k = _angle(z_safe)[:, None]  # phase_k = (B, 1)

        # print(f"{x[..., 0]=}")
        # print(f"{characters=}")
        # print(f"char * x = \n{characters * x[..., 0]}")
        # print(f"{z=}")
        # print(f"{phase_k=}")

        for module in self.SeqVX[:-1]:
            x = module(x)

        log_psi = self.SeqVX[-1](x)  # log_psi = (B, 1)
        log_psi_k = log_psi + 1j * phase_k

        min_infinity = jnp.array(
            [[jnp.finfo(log_psi.dtype).min + 0j]], dtype=jnp.complex128
        )
        min_infinity = jnp.broadcast_to(min_infinity, log_psi_k.shape)

        log_psi_k = jnp.where(
            (jnp.abs(z) < self.eps)[:, None],
            min_infinity,
            log_psi_k,
        )

        # print(f"{log_psi=}")
        # print(f"{log_psi_k=}")
        # print("\n")

        return log_psi_k
