import warnings

import netket as nk
import numpy as np
from scipy.sparse.linalg import eigsh

SPIN_OPS = {
    "X": nk.operator.spin.sigmax,
    "Y": nk.operator.spin.sigmay,
    "Z": nk.operator.spin.sigmaz,
}


class CouplingModel:
    """
    This module contains some base classes for models.

    In general, a model is expected to represent a hamiltonian of a system in terms
    of the operators which act in the atoms's Hilbert Space and also one could infer
    the many-body system wavefunction. With the Variational Ansatzs considered in
    this project no Hilbert space is holded, so has no capability of infer many
    characteristics from a quantum many-body system of spins.

    In this way, ``Models`` focuses on construct two main features of every quantum
    model: **onsite terms**, in which one can specify the field strength for each
    atom, and **coupling terms** which are a sequence of matrices, describing each
    the 2 atom interaction via some operator. Both have different shapes depending
    on the number of operators that are implied in a certain model.

    Also the models lays on a ``Lattice`` which specifies the geometry of the
    systems and are the starting point to perform this couplings construction, since
    many of the class methods used to build these features are based on ``Lattice``
    class methods to find atoms neighbors.

    These models serve as a prelude to find the ground state energy via minimizers.

    ---- Parameters ----

        - lattice: (``Lattice``) The lattice where the couplings are performed.
        - operators:(list of list of str) Operators involved in the model. First one
                    atom couplings list (fields) and then two atom couplings operators.

    ---- Attributes ----

        - onsite_terms: As default, it initializes a np.zeros((N,M)), where N is the
                        number of field operators and M is the number of atoms (lattice.Q).
        - coupling_terms: As default, it initializes a np.zeros((N,M,M)), where N is the
                        number of coupling operators and M is the number of atoms (lattice.Q).

    """

    def __init__(self, lattice, ops):

        self.lattice = lattice
        self.operators = ops

        N_ons = np.shape(self.operators[0])[0]
        N_coup = np.shape(self.operators[1])[0]
        M = lattice.Q

        self.onsite_terms = np.zeros((N_ons, M))
        self.coupling_terms = np.zeros((N_coup, M, M))

    def build_hamiltonian(self, hilbert=None, S_operators=False):
        """
        NetKet operator of the model:

            H = sum_a sum_i h^a_i s^a_i + sum_ab sum_{i<j} J^ab_ij s^a_i s^b_j

        with s = sigma (Pauli, S_operators=False) or s = S = sigma/2
        (S_operators=True: fields scaled by 1/2, couplings by 1/4).
        The model terms are not modified.
        """
        ops = [c for group in self.operators for op in group for c in op]
        if not all(c in SPIN_OPS for c in ops):
            raise ValueError(
                f"All operators must be {list(SPIN_OPS)} or pairs of them, "
                f"got {self.operators}"
            )

        if hilbert is None:
            hilbert = nk.hilbert.Spin(s=0.5, N=self.lattice.Q)
        onsite_scale, coupling_scale = (0.5, 0.25) if S_operators else (1.0, 1.0)
        onsite_terms = np.asarray(self.onsite_terms) * onsite_scale
        coupling_terms = np.asarray(self.coupling_terms) * coupling_scale

        H = 0.0 * SPIN_OPS["Y"](hilbert, 0)

        for o, op in enumerate(self.operators[0]):
            sigma = SPIN_OPS[op]
            for i in range(self.lattice.Q):
                H += onsite_terms[o][i] * sigma(hilbert, i)

        for o, op in enumerate(self.operators[1]):
            sigma_i, sigma_j = SPIN_OPS[op[0]], SPIN_OPS[op[1]]
            for i in range(self.lattice.Q):
                for j in range(i + 1, self.lattice.Q):
                    H += coupling_terms[o, i, j] * (
                        sigma_i(hilbert, i) @ sigma_j(hilbert, j)
                    )

        return H

    def exact_energy_lanczos(
        self, hilbert=None, k=1, eigenstates=False, which="SA", S_operators=False
    ):
        """Lowest k eigenvalues (and eigenvectors) of build_hamiltonian."""
        H = self.build_hamiltonian(hilbert, S_operators=S_operators).to_sparse()
        return eigsh(H, k=k, return_eigenvectors=eigenstates, which=which)

    def _initialize_onsite_terms(self, onsites):
        """
        Initializes onsite_terms automatically from onsites (field terms).
        One can specify either the constant strength of the field for all atoms
        with a float, or a list setting the field value for each atom.
        If all field strengths are constant, one must put an integer for each
        field operator in ops[0].

        On the other hand, one can specify the strength of fields for each atom
        in one or more field components just putting a list. In this regard,
        this method will return a regular Array for onsite_terms, repeating the
        strength for all atoms in the case the field is constant.

        Example: ops[0]=['X', 'Y', 'Z']          |
                 onsites=[1., [1., 2., 3.], 6.]  |

                 --> onsite_terms = np.array([[1., 1., 1.], [1., 2., 3.], [6., 6., 6.]])
        """

        shapes = [np.shape(term) for term in onsites]

        if any(shapes):

            N = np.shape(self.operators[0])[0]
            M = self.lattice.Q
            onsite_terms = []
            for i in range(N):

                if np.shape(onsites[i]):
                    onsite_terms.append(onsites[i])

                else:
                    onsite_terms.append([onsites[i]] * M)

            self.onsite_terms = np.array(onsite_terms)

        else:
            M = self.lattice.Q
            self.onsite_terms = [[onsite] * M for onsite in onsites]

    def add_onsite_all(self, strength, op):
        """
        Sets the same strength value of a field (op) for
        all atoms in lattice.
        Input:
            - strength:(float) strength of the field.
            - op:(str) Operator in self.operators for which is setting the coupling.

        Output:
            - Modified onsite_terms attribute.
        """
        op_idx = self.operators[0].index(op)
        onsite = [strength] * self.lattice.Q
        self.onsite_terms[op_idx] += np.array(onsite)

    def add_onsite_from_idx(self, strength, op, atom_idx):
        """
        Sets a strength field value of a field (op) for
        an atom from its index.
        Input:
            - strength:(float) strength of the field.
            - atom_idex:(int) Atom index in terms of self.lattice.order.
            - op:(str) Operator in self.operators for which is setting the coupling.

        Output:
            - Modified onsite_terms attribute.
        """

        op_idx = self.operators[0].index(op)
        self.onsite_terms[op_idx][atom_idx] += strength

    def add_onsite_from_latidx(self, strength, op, lat_idx):
        """

        Sets a strength field value of a field (op) for an atom from its lattice index.

        Input:
            - strength:(float) strength of the field.
            - lat_idx:(list) Lattice index of the atom ~ [u1, u2, c]
            - op:(str) Operator in self.operators for which is setting the coupling.

        Output:
            - Modified onsite_terms attribute.
        """

        op_idx = self.operators[0].index(op)
        atom = self.lattice.lat2idx([lat_idx])
        self.onsite_terms[op_idx][atom] += strength

    def add_couplings_from_dict(self, strength, op, neighbors, complementary=False):

        op_idx = self.operators[1].index(op)

        for i, neigh in neighbors.items():
            self.coupling_terms[op_idx, i, neigh] += strength
            if complementary:
                self.coupling_terms[op_idx, neigh, i] += strength

    def add_couplings_from_pairs(self, strength, op, pair, complementary=False):
        """

        Sets 2-atom couplings of an operator (op) automatically from a group of neighbors (pair)
        in self.lattice.pairs. Based on ``add_coupings_from_dict(self)``.

        Input:
            - strength:(float) strength of the field.
            - op:(str) Operator in self.operators for which is setting the coupling.
            - pair:(str) Group of neighbors in self.lattice.pairs ('NN'/'NN2'/'NN3'/...)
            - complementary:(bool) False as default. If True sets the transposed element
                            for each coupling.
        Output:
            - Modified coupling_terms attribute.
        """

        neighbors = self.lattice.find_neighbors_from_pair(pair)

        self.add_couplings_from_dict(strength, op, neighbors, complementary)

    def add_couplings_from_dx(self, strength, op, pair, dx_idx, complementary=False):
        """

        Sets 2-atom couplings of an operator (``op``) automatically from a ``dx`` vector which belongs
        to ``self.lattice.pairs[pair]``. Based on ``add_coupings_from_dict(self)``. It is used to
        perform couplings in anisotropic models.

        Input:
            - strength:(float) strength of the field.
            - op:(str) Operator in self.operators for which is setting the coupling.
            - pair:(str) Group of neighbors in self.lattice.pairs ('NN'/'NN2'/'NN3'/...)
            - dx_idx:(int) Index of the vector within ``self.lattice.pairs[pair]``
            - complementary:(bool) False as default. If True sets the transposed element
                            for each coupling.

        Output:
            - Modified ``self.coupling_terms`` attribute.
        """
        neighbors = self.lattice.find_neighbors_from_dx_pair(pair, dx_idx)
        self.add_couplings_from_dict(strength, op, neighbors, complementary)

    def add_coupling_term(self, strength, op, term, complementary=False):
        """
        Sets 2-atom couplings of an operator (op) from their lattice indices.

        Input:
            - strength:(float) strength of the field.
            - op:(str) Operator in ``self.operators`` for which is setting the coupling.
            - term:(tuple) Tuple of the two atom lattice indices for which the coupling
                    is set. Ex: term = ([0,0,1], [2,4,0])
            - complementary:(bool) False as default. If True sets the transposed element
                            for each coupling.
        Output:
            - Modified ``self.coupling_terms`` attribute.
        """

        assert all(
            lat_i in self.lattice.order for lat_i in term
        ), "Lat_index in term is out of lattice limits"

        op_idx = self.operators[1].index(op)
        i, j = self.lattice.lat2atom_idx(term)
        self.coupling_terms[op_idx, i, j] += strength
        if complementary:
            self.coupling_terms[op_idx, j, i] += strength

class IsingGeneral(CouplingModel):
    
    def __init__(self, lattice, J, fields= [0.0, 0.0, 0.0], ops=[["X", "Y", "Z"], ["ZZ"]]):

        _shape_control(fields, J, ops)

        super().__init__(lattice, ops)

        self._initialize_onsite_terms(fields)

        self.add_couplings_from_pairs(J, ops[1][0], "NN", complementary=True)



class LongRangeModel(CouplingModel):
    """ """

    def __init__(
        self,
        lattice,
        fields,
        hoppings,
        alpha,
        ops=[["X", "Z"], ["ZZ"]],
        norm_mode="min_norm",
    ):

        _shape_control(fields, hoppings, ops)

        super().__init__(lattice, ops)

        self._initialize_onsite_terms(fields)

        positions = self.lattice.calc_positions(
            self.lattice.order, self.lattice.basis, self.lattice.unit_cell_pos
        )

        distances = self.lattice.calc_distances(positions, self.lattice.bc)

        for i in range(len(self.coupling_terms)):

            self.coupling_terms[i] = self.long_range_couplings(
                hoppings[i], alpha, distances, norm_mode
            )

    def normalization(self, distances, alpha, mode):

        if self.lattice.bc == "open":

            if mode == "max_norm":
                norm = []

                for i in range(len(distances)):
                    norm.append(
                        1 + np.sum(distances[i][distances[i] != 0.0] ** (-alpha))
                    )

                norm = max(norm)

            elif mode == "min_norm":
                norm = 1 + np.sum(distances[0][distances[0] != 0.0] ** (-alpha))

            else:
                raise AttributeError(
                    f"No normalization mode ({mode}) for "
                    f"{self.lattice.bc} boundary conditions"
                )

        elif self.lattice.bc == "periodic":

            if mode == "min_norm":

                norm = 1 + np.sum(distances[0][distances[0] != 0.0] ** (-alpha))

            else:
                raise AttributeError(
                    f"No normalization mode ({mode}) for "
                    f"{self.lattice.bc} boundary conditions"
                )

        else:
            raise AttributeError(f"Error in lattice boundary conditions")

        return norm

    def long_range_couplings(self, J, alpha, distances, norm_mode):

        warnings.filterwarnings("ignore")
        J_matrix = np.where(distances != 0, J * distances ** (-alpha), 0.0)
        warnings.filterwarnings("default")

        norm = self.normalization(distances, alpha, norm_mode)

        return J_matrix / norm


class IsingChainXZ(CouplingModel):
    """
    Implementacion del modelo de Ising con campo transversal X y longitudinal Z
    con acoplos ZZ a primeros vecinos
    """

    def __init__(self, lattice, fields, hopping):

        ops = [["X", "Z"], ["ZZ"]]
        _shape_control(fields, hopping, ops)

        super().__init__(lattice, ops)

        self._initialize_onsite_terms(fields)

        self.add_couplings_from_pairs(hopping, "ZZ", "NN", complementary=True)


class GeneralNeighborCoupling(CouplingModel):
    """
    Generates a coupling model over any lattice through a tuple of the form:

    couplings = (strength, operator, neighbor_group) Ex: (0.2, 'ZZ', 'NN2')

    It also admits field terms:

    fields = (strength, operator)

    """

    def __init__(self, lattice, fields, couplings):

        fields = [field for field in fields if field[0] != 0]
        couplings = [coupling for coupling in couplings if coupling[0] != 0]

        ops_field = list(set([field[1] for field in fields]))
        ops_coupling = list(set([coupling[1] for coupling in couplings]))

        super().__init__(lattice, [ops_field, ops_coupling])

        for strength, op in fields:
            self.add_onsite_all(strength, op)

        for strength, op, neigh in couplings:
            self.add_couplings_from_pairs(strength, op, neigh, complementary=True)


class HeisenbergXYZ(CouplingModel):

    def __init__(
        self, lattice, hoppings, fields, ops=[["X", "Y", "Z"], ["XX", "YY", "ZZ"]]
    ):

        _shape_control(fields, hoppings, ops)

        super().__init__(lattice, ops)

        self._initialize_onsite_terms(fields)

        for hop, op in zip(hoppings, ops[1]):

            for J, pair in zip(hop, list(self.lattice.pairs.keys())):

                self.add_couplings_from_pairs(J, op, pair, complementary=True)


class JKGammaModel(CouplingModel):

    def __init__(self, lattice, couplings, direction_ops, field=[[], []]):

        coupling_ops = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]

        super().__init__(lattice, [field[1], coupling_ops])

        self._initialize_onsite_terms(field[0])

        J_list, K_list, Gamma_list = couplings

        # Isotropic Heisemberg interaction

        heisemberg_ops = ["XX", "YY", "ZZ"]

        for J, pair in zip(J_list, list(self.lattice.pairs.keys())):

            for op in heisemberg_ops:

                self.add_couplings_from_pairs(J, op, pair, complementary=True)

        # Kitaev interaction

        for K, Gamma, pair_ops, pair in zip(
            K_list, Gamma_list, direction_ops, list(self.lattice.pairs.keys())
        ):

            for dx_idx, dir_op in enumerate(pair_ops):

                self.add_couplings_from_dx(K, dir_op, pair, dx_idx, complementary=True)

                gamma_ops = [
                    op
                    for op in coupling_ops
                    if dir_op[0] not in op and op not in heisemberg_ops
                ]

                for gamma_op in gamma_ops:
                    self.add_couplings_from_dx(
                        Gamma, gamma_op, pair, dx_idx, complementary=True
                    )


class AnisotropicModel(CouplingModel):

    def __init__(self, lattice, coupling_operators, couplings, field=[[], []]):

        unraveled_ops = [
            op
            for pair_ops in coupling_operators
            for dx_ops in pair_ops
            for op in dx_ops
        ]

        super().__init__(lattice, [field[1], unraveled_ops])

        self._anisotropic_shape_control(couplings, coupling_operators, field)

        if field[0] != 0:
            self._initialize_onsite_terms(field[0])

        for pair, pair_ops, pair_coup in zip(
            self.lattice.pairs.keys(), coupling_operators, couplings
        ):

            for dx_idx, (dx_ops, dx_coup) in enumerate(zip(pair_ops, pair_coup)):

                for op, strength in zip(dx_ops, dx_coup):

                    self.add_couplings_from_dx(
                        strength, op, pair, dx_idx, complementary=True
                    )

    def _anisotropic_shape_control(self, couplings, coupling_operators, field):

        N_dx_in_pairs = [
            True if len(value) == len(couplings[i]) else False
            for i, value in zip(range(len(couplings)), self.lattice.pairs.values())
        ]
        ops_per_coupling = [
            True if len(coup) == len(ops) else False
            for coupling, operators in zip(couplings, coupling_operators)
            for coup, ops in zip(coupling, operators)
        ]

        if len(field[0]) != len(field[1]):
            raise ValueError(f"Both field components must have the same lenght")

        if not all(N_dx_in_pairs):
            raise ValueError(
                f"Lenght of couplings[i] must match the number of dx_vectors within"
                f"'lattice.pairs[i]'"
            )

        if not all(ops_per_coupling):
            raise ValueError(
                f"Shapes of 'couplings' and 'coupling_operators' must match. There must "
                f"be as many couplings as operators "
            )


def _shape_control(fields, hoppings, ops):

    if len(fields) != np.shape(ops[0])[0]:
        raise ValueError(
            f"Number of field elements in onsite_terms must fit"
            f"number of field operators, but are "
            f"{np.shape(fields)[0]} and {np.shape(ops[0])[0]}"
        )

    if len(hoppings) != np.shape(ops[1])[0]:
        raise ValueError(
            f"Number of hoppings must fit number of coupling operators"
            f" but are {np.shape(hoppings)[0]} and {np.shape(ops[1])[0]}"
        )
