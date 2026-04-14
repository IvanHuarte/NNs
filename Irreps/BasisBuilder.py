import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from typing import List

class DenseBasisBuilder:
    def __init__(self, symm_group, all_states):
        pass

    def unitary_representation(
        self, operation: callable, all_states: jax.typing.ArrayLike
    ) -> jax.typing.ArrayLike:
        """
        Calculates U(g) as the unitary representation of the symmetry operation g acting
        on the Hilbert space, which is a permutation matrix in the real-space basis.
        """
        permutations = []

        for state in all_states:
            permuted = operation(state)
            perm_idx = jnp.sum(jnp.abs(all_states - permuted[None, :]), axis=-1)
            perm_idx = jnp.nonzero(perm_idx == 0)[0]
            permutations.append(jax.nn.one_hot(perm_idx, all_states.shape[0]).squeeze())
        return jnp.asarray(permutations).T
    

class ActionBasisBuilder:
    def __init__(self, symm_group, all_states):
        pass

class OrbitBasisBuilder:
    def __init__(self, symm_group, all_states):
        pass