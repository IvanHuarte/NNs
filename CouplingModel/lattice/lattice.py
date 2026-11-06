import jax.numpy as jnp
import numpy as np

"""
Classes inspired in TeNPy 
"""


class Lattice:
    """

    A regular lattice. The lattice consists of a **unit cell** which is repeated
    `L=(L_u1, L_u2)` times in each `basis` direction. An atom of the lattice is
    described in terms of its **lattice indices** ``[u1, u2, c]``, where
    ``0 <= u_i < L[i] (i=1,2)`` are the unit cell cordinates respect to the lattice
    basis and ``0 <= c < Lu`` is the atom index within the unit cell. The atom is
    located in space via ``x = sum_i (u_i + unit_cell_pos[c][i]) * basis`` and is used
    in LongRangeModels to find neighbors and calculate coupling strenghts. It's also
    used in lattice plots.

    In addition to this, the lattice provides an **order** in which the lattice is
    mapped and which, in some sense, describes how close is the couplings topology               # REVISAR EL TRIPLAZO
    to a diagonal representation.

    Another feature of the lattice is the posibility to choose between ``open`` and
    ``periodic`` boundary conditions for the lattice. The differences lays on the
    calculations of distances between the atoms in the lattice, and the neighbors
    of atoms via neighbors_from_distances() and find_neighbors_from_pair().

    Finally, it includes a set of vectors to sistematically find the nearest-neighbors
    ('NN'), next-nearest neighbors('NN2'), next-next-nearest neighbors('NN3'), ... and on.
    This vectors (**pairs**) are organized in a dictionary whose keys are that strings
    specifying the neighborhood ('NN'/'NN2'/'NN3'/...) and the vectors as values.

    A ``pair`` vector establishes a conexion between two atoms in terms of their lattice
    indices and it is defined as follows:

                ``dx = (c_1, c_2, np.array(dx_u1, dx_u2))``

    , where c_1 and c_2 are the indices of both atoms within their unit cells. The last
    element specifies the displacement of the unit cell along the two basis vectors.

    ---- Parameters ----

        - L:(list of int) The lenght in each direction
        - basis:(list of 1D arrays) Lattice basis.
        - unit_cell_pos:(list) List in which each element is a tuple/list specifying
                            the cordinates of each atom within the unit cell.
        - bc:(str) Boundary conditions for the lattice.
        - pairs:(dict) Of the form:``{'NN':[dx_1, dx_2, ...], 'NN2':[dx_1, ...], ...}``

    ---- Attributes ----

        - L:(list of int) The lenght in each direction.
        - basis:(list of 1D arrays) Lattice basis.
        - unit_cell_pos:(list) List in which each element is a tuple/list specifying
                            the cordinates of each atom within the unit cell.
        - Lu:(int) Numer of atoms within the unit cell.
        - Q:(int) Total number of atoms in lattice.
        - bc:(str) Boundary conditions for the lattice.
        - pairs:(dict) Of the form:``{'NN':[dx_1, dx_2, ...], 'NN2':[dx_1, ...], ...}``

    """

    def __init__(
        self, L, basis, unit_cell_pos, bc="open", order="default_1", pairs=None
    ):

        self.L = L
        self.basis = basis
        self.unit_cell_pos = unit_cell_pos
        self.Lu = np.shape(unit_cell_pos)[0]
        self.Q = int(np.prod(L) * self.Lu)
        self.bc = bc
        self.order_mode = order
        self.pairs = pairs

        self.order = self.generate_lattice(self.L, self.Lu, self.order_mode)

        self.positions = self.calc_positions(self.order, basis, unit_cell_pos)
        # self.distances = self.calc_distances(self.positions, bc)
        self.neighbors = self.neighbors_from_positions(self.positions, NN_max=3)

    def generate_lattice(self, L, Lu, order):
        """
        It generates a list of lattice indices. This sequence forms the ``order`` in which
        the lattice is mapped.

        Inputs:
            -L:([int]) Size of the 2D lattice. 1D lattices are initialized as
                [L_u1,1]
            -Lu:(int) Number of atoms within the unit cell.

        Output:
            - order: ([[], ]) List of lattice indices for each atom.
        """
        if order == "default_1":
            return [
                [u1, u2, c]
                for u2 in range(L[1])
                for u1 in range(L[0])
                for c in range(Lu)
            ]

        elif order == "default_2":
            return [
                [u1, u2, c]
                for u1 in range(L[0])
                for u2 in range(L[1])
                for c in range(Lu)
            ]

        elif order == "snake_1":
            default_1 = [
                [u1, u2, c]
                for u2 in range(L[1])
                for u1 in range(L[0])
                for c in range(Lu)
            ]
            return [
                segment
                for i in range(L[1])
                for segment in default_1[i * L[0] : (i + 1) * L[0]][:: (-1) ** i]
            ]

        elif order == "snake_2":
            default_2 = [
                [u1, u2, c]
                for u1 in range(L[0])
                for u2 in range(L[1])
                for c in range(Lu)
            ]
            return [
                segment
                for i in range(L[0])
                for segment in default_2[i * L[1] : (i + 1) * L[1]][:: (-1) ** i]
            ]

        else:
            raise ValueError(
                f"Unknown order: {order}. Valid options are: 'default_1', 'default_2', 'snake_1', 'snake_2'"
            )

    def calc_positions(self, order, basis, uc_pos):
        """
        Calculates atom's positions in space from lattice cordinates. It's
        used to define distances between atoms and plotting the lattice.

        Inputs:
            -order: ([[], ]) List of lattice indices for each atom.
            -basis: (Array) Vectors that define in which directions the unit
                    cell is repeated.
            -uc_pos: Atoms cordinates within the unit cell.

        Output:
            -positions: (list of Array) List where each element is an array
                        np.array(x,y) defining its position in the XY plane.

        """

        return [
            (idx[0] + uc_pos[idx[2]][0]) * basis[0]
            + (idx[1] + uc_pos[idx[2]][1]) * basis[1]
            for idx in order
        ]

    def calc_distances(self, positions, bc):
        """
        It calculates distances between the atoms from their positions in the
        lattice. It has diferent behaviours whether the boundary conditions (bc)
        are open or periodic. If bc='periodic', it returns the minimum distance.

        Inputs:
            -positions: (list of Array) List where each element is an array
                        np.array(x,y) defining its position in the XY plane.
            -bc: (str) Boundary conditions for the lattice 'open'/'periodic'.

        Output:
            -distances: (Array()) Distances between atoms.
        """

        positions = jnp.array(positions, dtype="float64")
        Q = positions.shape[0]
        distances = jnp.zeros((Q, Q), dtype="float64")
        indices = jnp.triu_indices_from(distances, k=1)

        if bc == "open":

            dis = jnp.linalg.norm(positions[indices[0]] - positions[indices[1]], axis=1)

        elif bc == "periodic":

            Lu1 = self.L[0] * self.basis[0]
            Lu2 = self.L[1] * self.basis[1]
            direct_dist = jnp.linalg.norm(
                positions[indices[0]] - positions[indices[1]], axis=1
            )
            mirr_u1_dist = jnp.linalg.norm(
                Lu1 - jnp.abs(positions[indices[0]] - positions[indices[1]]), axis=1
            )
            mirr_u2_dist = jnp.linalg.norm(
                Lu2 - jnp.abs(positions[indices[0]] - positions[indices[1]]), axis=1
            )
            mirr_u1_u2_dist = jnp.linalg.norm(
                Lu1 + Lu2 - jnp.abs(positions[indices[0]] - positions[indices[1]]),
                axis=1,
            )
            dis = jnp.minimum(direct_dist, mirr_u1_dist)
            dis = jnp.minimum(dis, mirr_u2_dist)
            dis = jnp.minimum(dis, mirr_u1_u2_dist)

        distances = distances.at[indices].set(dis)
        distances += jnp.transpose(distances)

        return np.array(distances)

    def neighbors_from_distances(self, distances, NN_max=None, atol=1e-6):
        """
        Calculates the neighbors for all atoms in the lattice. It is organized
        as a dictionary where each item is defined as it follows:

        'atom_idx' : [[NN],[NN2],[NN3],...]

        , where atom_idx (int) is the atom index of the atom, following the ``order``.
        [NN],[NN2],... are sequences of atom indices which are nearest neighbors(NN),
        next-nearest neighbors(NN2), ...

        Input:
            -distances: (Array()) Distances between atoms.
            -NN_max: Number of nearest neighbors to calculate. If None(def) it calculates
                     all neighbors.
            -atol:(float) Absolute tolerance. Quantity from which two atoms belongs to the
                   same neighbor group.

        Output:
            -neighbors:(dict) Neighbors for all atoms in lattice.

        *** This function returns a complete dictionary of neighbors, i.e., for all
            'i' : [[],...,[j],...,[]] there exists its complementary
            'j' : [[],...,[i],...,[]]

        """
        neighbors = []

        for lat_dist in distances:

            v = np.sort(lat_dist)[1:]
            v_idx = np.argsort(lat_dist)[1:]
            mask = np.abs(v[1:] - v[:-1]) <= atol

            N = [[]]
            N[0].append(int(v_idx[0]))
            for i in range(len(mask)):

                if mask[i]:
                    N[-1].append(int(v_idx[i + 1]))
                else:
                    if len(N) == NN_max:
                        break
                    else:
                        N.append([int(v_idx[i + 1])])

            neighbors.append(N)

        return dict([(i, neighbors[i]) for i in range(len(neighbors))])

    def neighbors_from_positions(self, positions, NN_max=None):
        """
        Shortcut to calculate neighbors from their positions in the lattice.
        """

        distances = self.calc_distances(positions, self.bc)
        neighbors = self.neighbors_from_distances(distances, NN_max)

        return neighbors

    def terms_from_dx(self, dx, u_idx):
        """
        It recieves a ``dx`` vector and returns a list of all terms
        (p0,p1)=([u1_0, u2_0, c0], [u1, u2, c1]) that ``dx`` can generate
        from the unit cell cordinates (u_idx). Each term describes a conection
        between two atoms in the lattice.

        Input:
            - dx:(tuple) Element of self.pairs[pair]. It has the form
                of (c_1, c_2, np.array(dx_u1, dx_u2)) as described in ``Lattice``
                documentation.
            - u_idx:(array) Array of indices to iterate over u_1 and u_2.

        Output:
            -terms:(list of tuple) List of conections between atoms as described
                    above.
        """

        terms = []

        for u1_0, u2_0 in u_idx:

            c0 = dx[0]
            c1 = dx[1]
            u1 = int(u1_0 + dx[2][0])
            u2 = int(u2_0 + dx[2][1])
            terms.append(([int(u1_0), int(u2_0), c0], [u1, u2, c1]))

        return terms

    def find_terms_from_pair(self, pair):
        """
        Returns the terms for all pair vectors in pairs[pair]

        Input:
            -pair:(str) Element of self.pairs.keys(), which defines the neighbor
                        group. ('NN', 'NN2', 'NN3', ...)

        Output:
            -terms:(list of tuple) Tuples describing the conexions for the neighbor
                                   group. (p0,p1)=([u1_0, u2_0, c0], [u1, u2, c1])
        """
        delta_x = self.pairs[pair]
        u_idx = np.indices(self.L).transpose(1, 2, 0).reshape(-1, 2)

        terms = []

        for dx in delta_x:
            terms.append(self.terms_from_dx(dx, u_idx))

        terms = [term for term_dx in terms for term in term_dx]

        return terms

    def find_terms_of_dx_pair(self, dx):
        """
        Similar as ``find_terms_from_pair()``. It returns the terms for only one
        pair vector ``dx``.

        Input:
            -dx:(tuple) Of the form described in ``Lattice``.

        Output:
            -terms:(list of tuple) Tuples describing the conexions for the neighbor
                                   group. (p0,p1)=([u1_0, u2_0, c0], [u1, u2, c1])

        """

        u_idx = np.indices(self.L).transpose(1, 2, 0).reshape(-1, 2)

        terms = self.terms_from_dx(dx, u_idx)

        return terms

    def neighbors_from_terms(self, terms):
        """
        It returns a dictionary of the neighbors of the atoms specified in ``terms``.
        If lattice.bc='open' it will remove the terms (p0,p1) whose p1 point is out
        of the lattice limits. If lattice.bc = 'periodic' it will calculate its
        periodic_neighbor().

        Input:
            - terms:(list of tuple) Tuples describing the conexions for the neighbor
                                   group. (p0,p1)=([u1_0, u2_0, c0], [u1, u2, c1])

        Output:
            - neighbors:(dict)  Neighbors for all atoms in lattice.
        """

        neighbors = dict((i, []) for i in range(len(self.order)))

        if self.bc == "periodic":

            terms = [(p0, self.periodic_neighbor(p1)) for p0, p1 in terms]

        neigh = [self.lat2atom_idx(l_p) for l_p in terms]
        neigh = [n for n in neigh if not None in n]

        [neighbors[i].append(j) for i, j in neigh]

        if self.bc == "open":
            neighbors = dict(
                [(key, value) for key, value in neighbors.items() if np.shape(value)[0]]
            )

        return neighbors

    def find_neighbors_from_pair(self, pair):
        """
        Returns the neighbors of all atoms for all vectors of a neighbor group
        specified in ``pairs[pair]``.

        Input:
            -pair:(str) Element of self.pairs.keys(), which defines the neighbor
                        group ('NN', 'NN2', 'NN3', ...).

        Output:
            - neighbors:(dict)  Neighbors for all atoms in lattice.

        *** This function does NOT return a complete dictionary of neighbors, i.e.,
            for all 'i' : [[],...,[j],...,[]] its complementary
            'j' : [[],...,[i],...,[]] doesn't exists.
        """

        terms = self.find_terms_from_pair(pair)

        neighbors = self.neighbors_from_terms(terms)

        return neighbors

    def find_neighbors_from_dx_pair(self, pair, dx_idx):
        """
        Returns the neighbors of all atoms for a single vectors of a neighbor group
        specified in ``pairs[pair][dx_idx]``.

        Input:
            -pair:(str) Element of self.pairs.keys(), which defines the neighbor
                        group ('NN', 'NN2', 'NN3', ...).
            -dx_idx:(int) Index of the pair vector within ``pairs[pair]`` for which
                          the neighbors are going to be found.

        Output:
            - neighbors:(dict)  Neighbors for all atoms in lattice.

        *** This function does NOT return a complete dictionary of neighbors, i.e.,
            for all 'i' : [[],...,[j],...,[]] its complementary
            'j' : [[],...,[i],...,[]] doesn't exists.
        """
        dx = self.pairs[pair][dx_idx]
        terms = self.find_terms_of_dx_pair(dx)
        neighbors = self.neighbors_from_terms(terms)

        return neighbors

    def atom2lat_idx(self, idx):
        """
        Transforms atom indices into lattice indices via ``self.order``. If idx
        belongs to lattice, it returns its lattice index. Otherwise, it returns ``None``.

        Input:
            -idx: (int | list of int) Index or indices of the atoms in the lattice
                                      in terms of atom indices.

        Output:
            - List of lattice indices of the form ``[[u_1, u_2, c], ...]``
        """

        return [self.order[i] if i < len(self.order) else None for i in idx]

    def lat2atom_idx(self, idx):
        """
        Transforms lattice indices into atom indices via ``self.order``. If idx
        belongs to lattice, it returns its lattice index. Otherwise, it returns ``None``.

        Input:
            -idx: (latice_idx | list of lattice_idx) Index or indices of the atoms in
                                                    the lattice in terms of atom indices.

        Output:
            - List of atom indices
        """

        return [self.order.index(i) if i in self.order else None for i in idx]

    def periodic_neighbor(self, idx):

        return [idx[0] % self.L[0], idx[1] % self.L[1], idx[2]]

    def plot_lattice(self, NN_max=0):

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        cmap = mpl.cm.rainbow
        norm = plt.Normalize(vmin=0, vmax=self.Lu - 1)

        positions = self.calc_positions(self.order, self.basis, self.unit_cell_pos)

        x = np.transpose(positions)[0]
        y = np.transpose(positions)[1]

        _, ax = plt.subplots(figsize=np.array(self.L) * 1.5)

        ax.set_xlim(
            -1,
            self.L[0] * self.basis[0][0]
            + self.L[1] * self.basis[1][0]
            + max(self.unit_cell_pos[:, 0])
            + 1,
        )
        ax.set_ylim(
            -1,
            self.L[0] * self.basis[0][1]
            + self.L[1] * self.basis[1][1]
            + max(self.unit_cell_pos[:, 1])
            + 1,
        )

        if NN_max != 0:
            self.plot_neighbors(ax, NN_max)

        handles, labels = plt.gca().get_legend_handles_labels()
        lgnd = dict(zip(labels, handles))
        ax.legend(lgnd.values(), lgnd.keys(), loc="best")

        for i in range(self.Lu):
            ax.scatter(x[i :: self.Lu], y[i :: self.Lu], color=cmap(norm(i)), lw=4)

        for i in range(len(self.order)):
            ax.text(
                x[i] - 0.27,
                y[i] + 0.05,
                f"{i}",
                color="red",
                fontsize=3 * np.log(np.prod(self.L)),
            )

        ax.grid()
        plt.tight_layout()

    def plot_neighbors(self, ax, NN_max):

        if NN_max > 6:
            raise ValueError("Too much neighbors to plot. " "The limit is 6 neighbors.")

        linestyles = ["-", "--", "-.", "dotted", "dotted", "dotted"]
        colors = ["k", "r", "orange", "g", "b", "purple"]
        nearest_neighbours = ["NN", "NN2", "NN3", "NN4", "NN5", "NN6"]

        for i, nn in enumerate(nearest_neighbours[:NN_max]):

            terms = self.find_terms_from_pair(nn)

            edge_positions = np.array(
                [self.calc_positions(n, self.basis, self.unit_cell_pos) for n in terms]
            )

            for segment in edge_positions:
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    ls=linestyles[i],
                    lw=2 - i * 0.3,
                    color=colors[i],
                    alpha=1 - i * 0.2,
                    label=nn,
                )


class Triangular(Lattice):
    """
    Triangular regular lattice.

    Parameters:
        - L_u1:(int) Number of unit cells along the first basis vector.
        - L_u2:(int) Number of unit cells along the second basis vector.
        - a:(float) ``a=1.`` as default. Lenght of the triangle side which
                    lays over the first basis vector.
        - b:(float) ``b=1.`` as default. Lenght of the triangle side which
                    lays over the second basis vector.
        - bc(str) ``open`` as default. Boundary conditions for the lattice.

    """

    def __init__(
        self, L_u1, L_u2, a=1.0, b=1.0, bc="open", order="default_1", **kwargs
    ):

        basis = np.array([[a, 0.0], [0.5 * a, np.sqrt(b**2 - (a / 2) ** 2)]])
        unit_cell_pos = np.array([[0.0, 0.0]])

        NN = [
            (0, 0, np.array([1, 0])),
            (0, 0, np.array([0, 1])),
            (0, 0, np.array([-1, 1])),
        ]
        NN2 = [
            (0, 0, np.array([1, 1])),
            (0, 0, np.array([-1, 2])),
            (0, 0, np.array([2, -1])),
        ]
        NN3 = [
            (0, 0, np.array([2, 0])),
            (0, 0, np.array([0, 2])),
            (0, 0, np.array([-2, 2])),
        ]

        pairs = dict()
        pairs.setdefault("NN", NN)
        pairs.setdefault("NN2", NN2)
        pairs.setdefault("NN3", NN3)

        super().__init__([L_u1, L_u2], basis, unit_cell_pos, bc, order, pairs, **kwargs)


class Square(Lattice):
    """
    Square regular lattice.

    Parameters:
        - L_u1:(int) Number of unit cells along the first basis vector.
        - L_u2:(int) Number of unit cells along the second basis vector.
        - bc(str) ``open`` as default. Boundary conditions for the lattice.
    """

    def __init__(self, L_u1, L_u2, bc="open", order="default_1", **kwargs):

        basis = np.array([[1.0, 0.0], [0.0, 1.0]])
        unit_cell_pos = np.array([[0.0, 0.0]])

        NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))]
        NN2 = [(0, 0, np.array([1, 1])), (0, 0, np.array([1, -1]))]
        NN3 = [(0, 0, np.array([2, 0])), (0, 0, np.array([0, 2]))]

        pairs = dict()
        pairs.setdefault("NN", NN)
        pairs.setdefault("NN2", NN2)
        pairs.setdefault("NN3", NN3)

        super().__init__([L_u1, L_u2], basis, unit_cell_pos, bc, order, pairs)


class Chain(Lattice):
    """
    Chain lattice.

    Parameters:
        - L_u1:(int) Number of unit cells along the first basis vector.
        - bc(str) ``open`` as default. Boundary conditions for the lattice.
    """

    def __init__(self, size, bc="open", order="default_1", **kwargs):

        basis = np.array([[1.0, 0.0], [0.0, 0.0]])
        unit_cell_pos = np.array([[0.0, 0.0]])

        NN = [(0, 0, np.array([1, 0]))]
        NN2 = [(0, 0, np.array([2, 0]))]
        NN3 = [(0, 0, np.array([3, 0]))]

        pairs = dict()
        pairs.setdefault("NN", NN)
        pairs.setdefault("NN2", NN2)
        pairs.setdefault("NN3", NN3)

        super().__init__(size, basis, unit_cell_pos, bc=bc, order=order, pairs=pairs)


class Saw(Lattice):
    """
    Saw-shapped lattice. Mix of ``Chain`` and ``Triangular`` lattices.

    Parameters:
        - L_u1:(int) Number of unit cells along the first basis vector.
        - a:(float) ``a=1.`` as default. Lenght of the triangle side which
                    lays over the first basis vector.
        - b:(float) ``b=1.`` as default. Lenght of the triangle side which
                    lays over the second basis vector.
        - bc:(str) ``open`` as default. Boundary conditions for the lattice.
    """

    def __init__(self, L_u1, a=1.0, b=1.0, bc="open"):

        unit_cell_pos = np.array([[0.0, 0.0], [a / 2, np.sqrt(b**2 + (a / 2) ** 2)]])
        basis = a * np.array([[1.0, 0.0], [0.0, 1.0]])

        NN = [
            (0, 1, np.array([0, 0])),
            (1, 0, np.array([1, 0])),
            (0, 0, np.array([1, 0])),
        ]
        NN2 = [(1, 1, np.array([1, 0]))]
        NN3 = [(0, 1, np.array([1, 0])), (0, 1, np.array([1, 0]))]

        pairs = dict()
        pairs.setdefault("NN", NN)
        pairs.setdefault("NN2", NN2)
        pairs.setdefault("NN3", NN3)

        super().__init__([L_u1, 1], basis, unit_cell_pos, bc, pairs)
