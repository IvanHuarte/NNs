import jax.numpy as jnp
import jax
from functools import partial


class Z2:

    def __init__(self):

        self.dim = 1
        self.N = 2
        self.character = lambda q, n: jnp.exp(-1j * jnp.pi * q * n)

    def operation(self, configuration):
        "Invert all spin values"
        return -1.0 * configuration


class Traslation:

    def __init__(self, lattice_size, subgroup):

        self.lattice_size = lattice_size
        self.axis = subgroup

        self.dim = 1
        self.N = lattice_size[self.axis]
        self.character = lambda q, n: jnp.exp(2.0j * jnp.pi * q * n / self.N)

        assert self.N != 1, f"Group size too low. N = 1"

    @partial(jax.jit, static_argnums=0)
    def operation(self, configuration):
        "Translate a give spin configuration bz one position along an axis."
        configuration = configuration.reshape(self.lattice_size)
        configuration = jnp.roll(configuration, 1, axis=self.axis).reshape(-1)
        return configuration


def c4_operation(lattice_size):
    def _operation(configuration):
        configuration = configuration.reshape(lattice_size)
        configuration = configuration.T[::-1]
        return configuration.reshape(-1)

    return _operation


def c2_operation(lattice_size):
    def _operation(configuration):
        configuration = configuration.reshape(lattice_size)
        configuration = configuration.T[::-1].T[::-1]
        return configuration.reshape(-1)

    return _operation


class Cn:

    def __init__(self, lattice_size, n):
        """
        C_n rotation symmetry
        """

        self.lattice_size = lattice_size

        self.dim = 1
        self.N = n
        self.character = lambda q, m: jnp.exp(2.0j * jnp.pi * q * m / n)

        assert self.N != 1, f"Group size too low. N = 1"
        assert all(L != 1 for L in self.lattice_size)

        if n == 4:
            self.operation = c4_operation(lattice_size)

        if n == 2:
            self.operation = c2_operation(lattice_size)

class Refl_H:

    def __init__(self, lattice_size):
        """
        Reflection symmetry respect to the horizontal axis
        """

        self.lattice_size = lattice_size

        self.dim = 1
        self.N = 2
        self.character = lambda k, m: 1 if k == 0 else (-1) ** m

        assert self.N != 1, f"Group size too low. N = 1"
        assert all(L != 1 for L in self.lattice_size)

    def operation(self, configuration):
        return configuration.reshape(self.lattice_size)[::-1].reshape(-1)


class Refl_V:

    def __init__(self, lattice_size):
        """
        Reflection symmetry respect to the horizontal axis
        """

        self.lattice_size = lattice_size

        self.dim = 1
        self.N = 2
        self.character = lambda k, m: 1 if k == 0 else (-1) ** m

        assert self.N != 1, f"Group size too low. N = 1"
        assert all(L != 1 for L in self.lattice_size)

    def operation(self, configuration):
        return configuration.reshape(self.lattice_size)[:, ::-1].reshape(-1)


class Refl_D:

    def __init__(self, lattice_size):
        """
        Reflection symmetry respect to the horizontal axis
        """

        self.lattice_size = lattice_size

        self.dim = 1
        self.N = 2
        self.character = lambda k, m: 1 if k == 0 else (-1) ** m

        assert self.N != 1, f"Group size too low. N = 1"
        assert all(L != 1 for L in self.lattice_size)

    def operation(self, configuration):
        return configuration.reshape(self.lattice_size).T.reshape(-1)


class Refl_AD:

    def __init__(self, lattice_size):
        """
        Reflection symmetry respect to the horizontal axis
        """

        self.lattice_size = lattice_size

        self.dim = 1
        self.N = 2
        self.character = lambda k, m: 1 if k == 0 else (-1) ** m

        assert self.N != 1, f"Group size too low. N = 1"
        assert all(L != 1 for L in self.lattice_size)

    def operation(self, configuration):
        return configuration.reshape(self.lattice_size)[:, ::-1].T[:, ::-1].reshape(-1)
