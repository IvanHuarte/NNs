import numpy as np

from ..lattice.lattice import Chain, Square, Triangular
from .cm import AnisotropicModel, HeisenbergXYZ, IsingGeneral, LongRangeModel


class Oxalate:

    def __init__(self, size, couplings, spherical=True, field=[[], []], **kwargs):

        J, K, Gamma = self.parametrics(couplings, spherical)

        lattice = Triangular(size[0], size[1], **kwargs)

        couplings = [[[K, Gamma, Gamma], [K, Gamma, Gamma], [K, Gamma, Gamma]]]

        coupling_operators = [
            [["XX", "YZ", "ZY"], ["ZZ", "XY", "YX"], ["YY", "XZ", "ZX"]]
        ]

        # Kitaev interaction

        self.cm = AnisotropicModel(lattice, coupling_operators, couplings, field)

        # Isotropic Heisemberg interaction

        for op in ["XX", "YY", "ZZ"]:

            self.cm.add_couplings_from_pairs(J, op, "NN", complementary=True)

    def parametrics(self, couplings, spherical):

        if spherical:
            assert len(couplings) == 3, "3 parameters expected (a, theta, phi)"
            assert all(
                [coup < 360 and coup >= 0 for coup in couplings[1:]]
            ), "Angles must belong the interval [0,2*pi)"
            a, theta, phi = couplings

            theta = np.radians(theta)
            phi = np.radians(phi)

            J = a * np.sin(theta) * np.cos(phi)
            K = a * np.sin(theta) * np.sin(phi)
            Gamma = a * np.cos(theta)

        else:
            assert len(couplings) == 3, "3 parameters expected (J, K, Gamma)"
            J, K, Gamma = couplings

        return J, K, Gamma


class LRChain:
    """
    L: [int] length of the chain
    J: [float] hopping constant for ZZ operator
    alpha: [float] long range parameter
    fields: List[float, float]: Indicating both X and Z fields respectively.

    """

    def __init__(self, size, J, alpha, fields, ops=[["Y", "Z"], ["ZZ"]], **kwargs):

        lattice = Chain(size, **kwargs)

        hoppings = [J]
        self.cm = LongRangeModel(lattice, fields, hoppings, alpha, ops)


class LRSquare:
    """
    size: [int, int] size of the square lattice
    J: [float] hopping constant for ZZ operator
    alpha: [float] long range parameter
    fields: List[float, float]: Indicating both X and Z fields respectively.
    kwargs: additional arguments for the lattice
    """

    def __init__(self, size, J, alpha, fields, ops=[["Y", "Z"], ["ZZ"]], **kwargs):

        lattice = Square(size[0], size[1], **kwargs)

        hoppings = [J]
        self.cm = LongRangeModel(lattice, fields, hoppings, alpha, ops)


class J1J2Square:
    """
    Implementation of J1-J2 model, an isotropic heisemberg interaction in
    (next) nearest-neighbors via J1 (J2).
    Inputs:
        size: [int, int] size of the square lattice
        J1: [float] hopping constant for NN
        J1: [float] hopping constant NN2
        fields: List[float, float, float]: Indicating both X and Z fields respectively.
        kwargs: additional arguments for lattice
    """

    def __init__(self, size, J1, J2, fields=[0.0, 0.0, 0.0], **kwargs):

        self.J1 = J1
        self.J2 = J2
        lattice = Square(size[0], size[1], **kwargs)

        hoppings = [
            [J1, J2],
            [J1, J2],
            [J1, J2],
        ]  # [[J1xx, J2xx], [J1yy, J2yy], [J1zz, J2zz]]

        self.cm = HeisenbergXYZ(lattice, hoppings, fields)


class IsingSquare:
    """
    Implementation of J1-J2 model, an isotropic heisemberg interaction in
    (next) nearest-neighbors via J1 (J2).
    Inputs:
        size: [int, int] size of the square lattice
        J1: [float] hopping constant for NN
        J1: [float] hopping constant NN2
        fields: List[float, float, float]: Indicating both X and Z fields respectively.
        kwargs: additional arguments for lattice
    """

    def __init__(self, size, J, fields=[0.0], ops=[["X"], ["ZZ"]], **kwargs):

        self.J = J
        lattice = Square(size[0], size[1], **kwargs)
        self.cm = IsingGeneral(lattice, [J], fields, ops)


class SSMSquare:
    """
    Implementation of Shastry-Sutherland model in a rectangular, an isotropic heisemberg interaction in
    nearest-neighbors via J1 and anisotropic connections via J2. The later
    Inputs:
        size: [int, int] size of the square lattice
        J: [float] hopping constant for NN
        Jp: [float] hopping constant NN2
        fields: List[float, float, float]: Indicating both X and Z fields respectively.
        kwargs: additional arguments for lattice
    """

    def __init__(self, size, J, Jp, fields=[0.0], **kwargs):

        self.J = J
        self.Jp = Jp
        lattice = Square(size[0], size[1], **kwargs)

        hoppings = [
            [J],
            [J],
            [J],
        ]  # [[J1xx, J2xx], [J1yy, J2yy], [J1zz, J2zz]]

        self.cm = HeisenbergXYZ(lattice, hoppings, fields)

        for u1, u2, c in lattice.order:
            if u2 % 2 == 0:
                if u1 % 2 == 0:

                    for op in ["XX", "YY", "ZZ"]:
                        self.cm.add_coupling_term(
                            Jp,
                            op,
                            ([u1, u2, 0], [(u1 + 1) % size[0], (u2 + 1) % size[1], 0]),
                            complementary=True,
                        )

                else:

                    for op in ["XX", "YY", "ZZ"]:
                        self.cm.add_coupling_term(
                            Jp,
                            op,
                            ([u1, u2, 0], [(u1 + 1) % size[0], (u2 - 1) % size[1], 0]),
                            complementary=True,
                        )
