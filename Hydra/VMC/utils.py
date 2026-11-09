import jax
from netket.optimizer import solver

# def solver_wrapper

solver_dict = {
    "pinv": solver.pinv,
    "pinv_smooth": solver.pinv_smooth,
    "cholesky": solver.cholesky,
    "cholesky_with_fallback": solver.cholesky_with_fallback,
    "LU": solver.LU,
    "cg": jax.scipy.sparse.linalg.cg,
    "gmres": jax.scipy.sparse.linalg.gmres,
    "bicgstab": jax.scipy.sparse.linalg.bicgstab,
}
