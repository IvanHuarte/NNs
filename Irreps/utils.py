import numpy as np
import jax.numpy as jnp
from itertools import product
from typing import List


def _all_idx_combinations(N: List):

    for idx in product(*(range(d) for d in N)):
        yield idx

def estimate_dim_q(projector, D, n_vecs=500):
    acc = 0.0
    for _ in range(n_vecs):
        v = np.random.choice([1, -1], size=D) + 0j
        acc += np.real(np.vdot(v, projector(v)))
    return int(round(acc / n_vecs))

def build_irrep_basis(
    projector,
    D,
    max_trials=5000,
    tol_new=1e-9,
    patience=30,
):
    """
    Construye una base ortonormal del subespacio Im(P_q)
    usando únicamente la acción del proyector.

    Devuelve Q con shape (D, dim_irrep),
    donde dim_irrep emerge automáticamente.
    """

    Q = []                      
    failures_in_a_row = 0
    last_dim = 0

    for i in range(max_trials):
        print(f"Step {i}")

        # 1) vector aleatorio
        v = np.random.randn(D) + 1j * np.random.randn(D)

        # 2) proyección
        w = projector(v)
        nrm = np.linalg.norm(w)

        if nrm < tol_new:
            failures_in_a_row += 1
            continue

        w /= nrm

        # 3) Gram–Schmidt
        for q in Q:
            w -= np.vdot(q, w) * q

        nrm = np.linalg.norm(w)
        if nrm > tol_new:
            Q.append(w / nrm)
            failures_in_a_row = 0
        else:
            failures_in_a_row += 1

        # 4) criterio de saturación (USO REAL DE tol_stop)
        if failures_in_a_row >= patience:
            # comprobamos que la dimensión ya no cambia
            if abs(len(Q) - last_dim) <= 0:
                break
            last_dim = len(Q)
            failures_in_a_row = 0

    if len(Q) == 0:
        raise RuntimeError("No se pudo construir ningún vector en la irrep")

    # 5) Re-ortonormalización global (imprescindible)
    Q = jnp.column_stack(Q)
    Q, _ = jnp.linalg.qr(Q)

    return Q


def projected_hamiltonian(H, Qq):
    """
    Construye H_q = Q† H Q usando LocalOperatorJax
    sin densificar H.
    """
    _, dim_q = Qq.shape
    Hq = jnp.zeros((dim_q, dim_q), dtype=jnp.complex128)

    for j in range(dim_q):
        Hj = H @ Qq[:, j]          # LocalOperatorJax aplicado a un vector
        Hq = Hq.at[:, j].set(jnp.conj(Qq).T @ Hj)

    return Hq


from scipy._lib._util import _apply_over_batch
from scipy.sparse.linalg import svds

@_apply_over_batch(('A', 2))
def svds_orth(A, rcond=None, k=6, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True,
         solver='arpack', rng=None, options=None):
    """
    Construct an orthonormal basis for the range of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Q : (M, K) ndarray
        Orthonormal basis for the range of A.
        K = effective rank of A, as determined by rcond

    See Also
    --------
    svd : Singular value decomposition of a matrix
    null_space : Matrix null space

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import orth
    >>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
    >>> orth(A)
    array([[0., 1.],
           [1., 0.]])
    >>> orth(A.T)
    array([[0., 1.],
           [1., 0.],
           [0., 0.]])

    """
    u, s, vh = svds(A, k=k, ncv=ncv, tol=0, which=which, v0=v0,
         maxiter=maxiter, return_singular_vectors=return_singular_vectors,
         solver=solver, rng=rng, options=options
    )
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s, initial=0.) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = u[:, :num]
    return Q