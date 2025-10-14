# %%
import enum

import jax
import jax.nn
import jax.numpy as jnp
import netket as nk
import numpy as onp
import numpy.linalg
import scipy as sp
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt


# %%
N_SPINS = 6
ISING_H = 2.0


class Z2_Irrep(enum.Enum):
    TRIVIAL = 0
    NONTRIVIAL = 1


# %%
graph = nk.graph.Hypercube(length=N_SPINS, n_dim=1, pbc=True)
hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
# hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=ISING_H)
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=graph, J=ISING_H)

# %%
hamiltonian_matrix = hamiltonian.to_dense()

# %%
configurations = hilbert.all_states()

# %%
# Build the permutation matrices corresponding to the two generators:
# translation by 1 and total flip.
n_configurations = configurations.shape[0]


def create_permutation_matrix(operation: callable):
    "Create the permutation matrix for a given Python function (a symmetry)."
    permutations = []
    for i_conf in range(n_configurations):
        conf = configurations[i_conf, :]
        translated = operation(conf)
        match = jnp.squeeze(
            jnp.nonzero(
                (configurations == translated[jnp.newaxis, :]).all(axis=1)
            )[0]
        )
        permutations.append(jax.nn.one_hot(match, n_configurations))
    return jnp.asarray(permutations).T


translation_matrix = create_permutation_matrix(lambda x: jnp.roll(x, 1))
inversion_matrix = create_permutation_matrix(lambda x: -x)

# %%
# Extract the symmetry-adapted bases for each irrep.


def create_projector(q_index: int, z2_irrep: Z2_Irrep):
    "Create an unnormalized projection operator for a given irrep."
    if q_index < 0 or q_index >= N_SPINS:
        raise ValueError("q-point index out of range")

    nruter = jnp.zeros((n_configurations, n_configurations), dtype=complex)

    operation_matrix = jnp.eye(n_configurations)
    for n_trans in range(N_SPINS):
        operation_matrix @= translation_matrix
        character = jnp.exp(-1.0j * 2.0 * jnp.pi * q_index * n_trans / N_SPINS)
        for _ in range(2):
            nruter += character * operation_matrix
            if z2_irrep == Z2_Irrep.NONTRIVIAL:
                character *= -1.0
            operation_matrix @= inversion_matrix

    return nruter


bases = {}
for q_index in range(N_SPINS):
    for z2_irrep in Z2_Irrep:
        projector = create_projector(q_index, z2_irrep)
        bases[(q_index, z2_irrep)] = sp.linalg.orth(projector)

# %%
# Summarize the results.
print("Dimensions of the bases:")
for irrep in bases:
    print(f"{irrep}:\t{bases[irrep].shape[1]}")

print("TOTAL:", sum(bases[irrep].shape[1] for irrep in bases))

# %%
# Build the joint basis.
adapted_basis = onp.concatenate(list(bases.values()), axis=1)
print("RANK OF THE BASIS MATRIX:", onp.linalg.matrix_rank(adapted_basis))

# %%
# Transform the Hamiltonian to the symmetry-adapted basis.
adapted_matrix = adapted_basis.conj().T @ hamiltonian_matrix @ adapted_basis

# %%
# Visually check if the matrix is block-diagonal.
abs_matrix = onp.abs(adapted_matrix)

plt.matshow(abs_matrix)
plt.colorbar()
count = 0
for irrep in list(bases.keys())[:-1]:
    count += bases[irrep].shape[1]
    plt.axhline(count - 0.5, color="white")
    plt.axvline(count - 0.5, color="white")
plt.title("Symmetry-adapted Hamiltonian")
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()
plt.show()

# %%
# Extract the eigenvalues block by block.
plt.figure()

global_min = onp.inf
global_max = -onp.inf
for i_z2_irrep, z2_irrep in enumerate(Z2_Irrep):
    plt.subplot(1, 2, i_z2_irrep + 1)
    for q_index in range(N_SPINS):
        basis = bases[(q_index, z2_irrep)]
        block = basis.conj().T @ hamiltonian_matrix @ basis
        eigvals = sp.linalg.eigvalsh(block)
        global_min = min(global_min, min(eigvals))
        global_max = max(global_max, max(eigvals))
        plt.scatter(
            [q_index] * eigvals.shape[0],
            eigvals,
            c=list(range(eigvals.shape[0])),
        )
    plt.title(z2_irrep)
    plt.xlabel("q")
    plt.ylabel("E")

plt.subplot(1, 2, 1)
plt.ylim(global_min, global_max)
plt.subplot(1, 2, 2)
plt.ylim(global_min, global_max)

plt.tight_layout()
plt.show()

# %%