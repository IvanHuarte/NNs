from typing import Tuple, List
import scipy as sp
import jax
import jax.numpy as jnp
import netket as nk
from Irreps.SymmBuilder import SymmGroup
from Irreps.utils import _all_idx_combinations

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
    

class OrbitDiagonalizer:
    def __init__(self, symm_group: SymmGroup, hilbert: nk.hilbert):
        self.symm_group = symm_group
        self.hilbert = hilbert
        self.g_dimensions = [subgroup.N for subgroup in self.symm_group.group]
        self.g_characters = self.group_character(self.symm_group.group)

        print("Computing group generators...")
        self.generators = self.get_idx_generators(symm_group.group, hilbert)

        print("Computing group orbits...")
        self.group_orbits = self.get_group_orbits(self.generators, hilbert.n_states, self.g_dimensions)

    def group_character(self, group):
        character = [symm.character for symm in group]
        def nruter(q,n):
            return jnp.array([char(q_i, n_i) for char, q_i, n_i in zip(character, q, n)], dtype=jnp.complex128).prod()
        return nruter        

    def get_idx_generators(self, group, hilbert):

        generators = []
        for symmetry in group:
            generator = hilbert.states_to_numbers(
                jax.vmap(lambda state: symmetry.operation(state))(hilbert.all_states())
            )
            generators.append(generator)

        return generators
    
    def power_generator(self, gen, idx, power):
        for _ in range(power):
            idx = int(gen[idx])
        return idx

    def get_group_orbits(self, generators, hilbert_dim: int, g_dimensions: Tuple[int]):

        orbits = []
        visited = set()

        for rep in range(hilbert_dim):
            if rep in visited:
                continue

            orbit = []

            for n_vector in _all_idx_combinations(g_dimensions):
                new_rep = rep
                for gen, n in zip(generators, n_vector):
                    new_rep = self.power_generator(gen, new_rep, n)
                orbit.append(new_rep)
                visited.add(new_rep)

            orbits.append(tuple(orbit))
        return orbits
    
    def build_irrep_basis(self, q_vector):

        q_basis = []

        n_vector = list(_all_idx_combinations(self.g_dimensions))
        # print(f"Building irrep basis for q_vector: {q_vector}...")

        for orbit in self.group_orbits:
            # print(f"Orbit: {orbit}")
            stabilizer_ns = [
                n_vector[k]
                for k, j in enumerate(orbit)
                if j == orbit[0] 
            ]

            # Stabilizer control
            if stabilizer_ns:
                # print(f"Stabilizer n_vector: {stabilizer_ns} ")
                    
                total = 0.0 + 0.0j
                for n in stabilizer_ns:
                    total += self.g_characters(q_vector, n)
                # print(f"Stabilizer control total: {total}")
                if jnp.abs(total) < 1e-12:
                    # print(f"Orbit is not in the irrep, skipping...\n")
                    continue
                # else:
                #     print("\n")

            # If the orbit contributes, we can build the basis vector

            state = {}
            for k, j in enumerate(orbit):
                n = n_vector[k]               
                state[j] = self.g_characters(q_vector, n)

            
            norm = jnp.sqrt(sum(jnp.abs(c)**2 for c in state.values()))
            for j in state:
                state[j] /= norm

            q_basis.append(state)

        return q_basis
    

    def apply_H_to_basis_index(self, H, j):
        # índice → configuración
        cfg = self.hilbert.numbers_to_states(int(j))

        # acción local del Hamiltoniano
        xs, mels = H.get_conn(cfg)

        out = {}
        for cfg_p, mel in zip(xs, mels):
            j_p = int(self.hilbert.states_to_numbers(cfg_p).item())
            out[j_p] = out.get(j_p, 0.0) + mel

        return out



    def apply_local_operator(self, H, state):
        """
        state: dict {hilbert_index: amplitude}
        """
        out = {}

        for j, amp in state.items():
            transitions = self.apply_H_to_basis_index(H, j)
            for jp, hval in transitions.items():
                out[jp] = out.get(jp, 0.0) + hval * amp

        return out



    def build_H_q_sparse(self, H, basis_q):
        dim_q = len(basis_q)
        H_q = jnp.zeros((dim_q, dim_q), dtype=jnp.complex128)

        # Precomputamos H |ψ_j⟩ en forma dispersa
        Hpsi = [
            self.apply_local_operator(H, state_j)
            for state_j in basis_q
        ]

        for i, psi_i in enumerate(basis_q):
            for j, Hpsi_j in enumerate(Hpsi):
                val = 0.0 + 0.0j
                # producto interno disperso
                for k, amp_i in psi_i.items():
                    if k in Hpsi_j:
                        val += jnp.conj(amp_i) * Hpsi_j[k]
                H_q = H_q.at[i, j].set(val)

        return H_q


    def diagonalize_by_blocks(
        self,
        hamiltonian, 
        q_vectors: List[Tuple[int, ...]] | None = None, 
        sanity_check=False
    ):

        eigvals = {}
        eigvecs = {}
        
        q_vectors = q_vectors if q_vectors is not None else _all_idx_combinations(self.g_dimensions)

        basis = {}
        for q_vector in q_vectors:
            print(f"Diagonalizing for q_vector: {q_vector}...")
            print(f"Building irrep basis...")
            basis = self.build_irrep_basis(q_vector)
            print(f"Basis size for q_vector {q_vector}: {len(basis)}")
            if sanity_check:
                print(f"\nPerforming sanity check for q_vector {q_vector}...\n")
                self.sanity_check(basis)

            print(f"Building H_q")
            H_q = self.build_H_q_sparse(hamiltonian, basis)
            print(f"H_q shape: {H_q.shape}")
            print(f"Diagonalizing H_q...")
            evals, evecs = sp.linalg.eigh(H_q)
            print(f"Diagonalization complete for q_vector\n")
            print(f"Eigenvalues for q_vector {q_vector}: {evals[:5]}...\n")
            eigvals[q_vector] = evals
            eigvecs[q_vector] = evecs

        return eigvals, eigvecs
    
    def sanity_check(self, basis):
        # Sanity check
        for q_vector, basis in basis.items():
            print(f"q_vector: {q_vector}, basis size: {len(basis)}")

            for state in basis:
                assert jnp.allclose(sum(abs(c)**2 for c in state.values()), 1.0)