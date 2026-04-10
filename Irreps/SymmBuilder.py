import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from typing import List
from itertools import product

from Irreps.Symmetry import SIMMETRY, EXTERNAL_ARGS


def _all_idx_combinations(N: List):

    for idx in product(*(range(d) for d in N)):
        yield idx


class SymmGroup:

    def __init__(self, setup, **kwargs):

        self.group, self.group_label = self._initialize_symmetries(setup, **kwargs)

        self.N_group = [symmetry.N for symmetry in self.group]
        self.norm = self._group_norm()

    def _initialize_symmetries(self, setup, **kwargs):

        group = []
        group_label = []

        for symm_name, symm_setup in setup.items():

            if symm_setup["on"] == False:
                continue

            init_kwargs = dict(
                (k, v) for k, v in symm_setup.items() if k not in ["on", "subgroup"]
            )

            for arg_name, arg_list in EXTERNAL_ARGS.items():
                if symm_name in arg_list:
                    init_kwargs[arg_name] = kwargs[arg_name]

            if "subgroup" in symm_setup.keys():
                for subg in symm_setup["subgroup"]:
                    group.append(SIMMETRY[symm_name](**init_kwargs, subgroup=subg))
                    group_label.append(f"{symm_name}_{subg}")
            else:
                group.append(SIMMETRY[symm_name](**init_kwargs))
                group_label.append(f"{symm_name}")

        return group, group_label

    def _unitary_representation(
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

    def _character(self, q_vector: List[int,], n_vector: List[int,]) -> complex:

        character = 1.0
        for i, symmetry in enumerate(self.group):
            character *= symmetry.character(q_vector[i], n_vector[i])
        return character

    def _group_norm(self):

        norm = 1.0
        for symmetry in self.group:
            norm *= symmetry.N
        return norm

    def get_unitary_representations(self, all_states: jax.typing.ArrayLike) -> List:
        """
        Returns the unitary representations for all the symmetries in the group.
        """
        representations = []
        for symmetry in self.group:
            representations.append(
                self._unitary_representation(symmetry.operation, all_states)
            )
        return representations

    def get_irrep_projector(
        self,
        representations: jax.typing.ArrayLike,
        q_vector: List[int,],
        norm: bool = False,
    ) -> jax.typing.ArrayLike:
        assert len(q_vector) == len(representations) == len(self.group)

        dim = len(representations[0])
        nruter = jnp.zeros((dim, dim), dtype=complex)

        for n_vector in _all_idx_combinations(self.N_group):

            character = self._character(q_vector, n_vector)
            nruter_2 = jnp.eye(dim)
            for i, representation in enumerate(representations):
                nruter_2 @= jla.matrix_power(representation, n_vector[i])

            nruter += character * nruter_2

        # TODO: Implement dimension for non-abelian groups

        if norm:
            nruter /= self.norm

        return nruter
