#!/home/ihuarte/miniconda3/envs/conda_env/bin/python

#  %%
import enum

import jax
import jax.nn
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.typing
import netket as nk
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 30

# %%
N_A = 3
N_B = 3
JS = [1.0, 0.5]


class Z2_Irrep(enum.Enum):
    TRIVIAL = 0
    SIGN = 1


# %%
edges = []
for i_a in range(N_A):
    for j_a in range(N_B):
        center = np.ravel_multi_index((i_a, j_a), (N_A, N_B))
        right_1 = np.ravel_multi_index(((i_a + 1) % N_A, j_a), (N_A, N_B))
        right_2 = np.ravel_multi_index(((i_a + 2) % N_A, j_a), (N_A, N_B))
        up_1 = np.ravel_multi_index((i_a, (j_a + 1) % N_B), (N_A, N_B))
        up_2 = np.ravel_multi_index((i_a, (j_a + 2) % N_B), (N_A, N_B))
        edges.append([center, right_1, 0])
        edges.append([center, right_2, 1])
        edges.append([center, up_1, 0])
        edges.append([center, up_2, 1])
graph = nk.graph.Graph(edges=edges)

# %%
hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=graph, J=JS)

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
            jnp.nonzero((configurations == translated[jnp.newaxis, :]).all(axis=1))[0]
        )
        permutations.append(jax.nn.one_hot(match, n_configurations))
    return jnp.asarray(permutations).T


def translate_by_one(configuration: jax.typing.ArrayLike, axis: int):
    "Translate a give spin configuration bz one position along an axis."
    configuration = configuration.reshape((N_A, N_B))
    configuration = jnp.roll(configuration, 1, axis=axis)
    return configuration.ravel()


translation_matrix_a = create_permutation_matrix(lambda x: translate_by_one(x, 0))
translation_matrix_b = create_permutation_matrix(lambda x: translate_by_one(x, 1))
inversion_matrix = create_permutation_matrix(lambda x: -x)

# %%
# Extract the symmetry-adapted bases for each irrep.

def create_projector(q_a: int, q_b: int, z2_irrep: Z2_Irrep):
    "Create an unnormalized projection operator for a given irrep."
    if q_a < 0 or q_a >= N_A or q_b < 0 or q_b >= N_B:
        raise ValueError("q-point index out of range")

    nruter = jnp.zeros((n_configurations, n_configurations), dtype=complex)

    for n_trans_a in range(N_A):
        for n_trans_b in range(N_B):
            character = jnp.exp(
                -2.0j * jnp.pi * (q_a * n_trans_a / N_A + q_b * n_trans_b / N_B)
            )
            operation_matrix = jla.matrix_power(
                translation_matrix_a, n_trans_a
            ) @ jla.matrix_power(translation_matrix_b, n_trans_b)
            nruter += character * operation_matrix
            if z2_irrep == Z2_Irrep.SIGN:
                character *= -1.0
            nruter += character * (operation_matrix @ inversion_matrix)

    return nruter


bases = {}
for q_a in range(N_A):
    for q_b in range(N_B):
        for z2_irrep in Z2_Irrep:
            projector = create_projector(q_a, q_b, z2_irrep)
            bases[(q_a, q_b, z2_irrep)] = sp.linalg.orth(projector)

# %%
# Summarize the results.
print("Dimensions of the bases:")
for irrep in bases:
    print(f"{irrep}:\t{bases[irrep].shape[1]}")

print("TOTAL:", sum(bases[irrep].shape[1] for irrep in bases))

# %%
# Build the joint basis.
adapted_basis = np.concatenate(list(bases.values()), axis=1)
print("RANK OF THE BASIS MATRIX:", np.linalg.matrix_rank(adapted_basis))

# %%
# Transform the Hamiltonian to the symmetry-adapted basis.
adapted_matrix = adapted_basis.conj().T @ hamiltonian_matrix @ adapted_basis

# %%
# Visually check if the matrix is block-diagonal.
abs_matrix = np.abs(adapted_matrix)

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

# %%
# Extract the eigenvalues block by block.
plt.figure(figsize=(20, 20))

eigvals = {}

global_min = np.inf
global_max = -np.inf
for i_z2_irrep, z2_irrep in enumerate(Z2_Irrep):
    plt.subplot(1, 2, i_z2_irrep + 1)
    for q_a in range(N_A):
        for q_b in range(N_B):
            basis = bases[(q_a, q_b, z2_irrep)]
            block = basis.conj().T @ hamiltonian_matrix @ basis
            eigvals[(q_a, q_b, z2_irrep)] = sp.linalg.eigvalsh(block)
            global_min = min(global_min, eigvals[(q_a, q_b, z2_irrep)].min())
            global_max = max(global_max, eigvals[(q_a, q_b, z2_irrep)].max())
            plt.scatter(
                [q_a] * eigvals[(q_a, q_b, z2_irrep)].shape[0],
                eigvals[(q_a, q_b, z2_irrep)],
                c=f"C{q_b}",
                label=f"$q_b = {q_b}$" if q_a == 0 else "",
                marker=q_b,
                alpha=0.5,
                s=100,
            )
    plt.title(str(z2_irrep))
    plt.xlabel("q")
    plt.ylabel("E")
    plt.legend(loc="best")

span = global_max - global_min
plt.subplot(1, 2, 1)
plt.ylim(global_min - 0.2 * span, global_max + 0.2 * span)
plt.subplot(1, 2, 2)
plt.ylim(global_min - 0.2 * span, global_max + 0.2 * span)

print("(0, 0, TRIVIAL):")
print(eigvals[0, 0, Z2_Irrep.TRIVIAL])
print("(0, 0, SIGN):")
print(eigvals[0, 0, Z2_Irrep.SIGN])

plt.tight_layout()
plt.show()

# %%
