import jax
from netket.optimizer import solver

solver_dict = {
    "pinv": solver.pinv,
    "pinv_smooth": solver.pinv_smooth,
    "cholesky": solver.cholesky,
    "LU": solver.LU,
    "cg": jax.scipy.sparse.linalg.cg,
    "gmres": jax.scipy.sparse.linalg.gmres,
    "bicgstab": jax.scipy.sparse.linalg.bicgstab,
}
