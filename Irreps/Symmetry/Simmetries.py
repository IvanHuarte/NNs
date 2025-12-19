import jax.numpy as jnp


class Z2():

    def __init__(self):

        self.dim = 1
        self.N = 2
        self.character = lambda q, n: jnp.exp(-1j* jnp.pi* q * n)

    def operation(self, configuration):
        "Invert all spin values"
        return -1.0 * configuration
    

class Traslation():

    def __init__(self, lattice_size, subgroup):

        self.lattice_size = lattice_size
        self.axis = subgroup

        self.dim = 1
        self.N = lattice_size[self.axis]
        self.character = lambda q, n: jnp.exp(-2j* jnp.pi* q * n / self.N)

        assert self.N != 1, f"Group size too low. N = 1"

    def operation(self, configuration):
        "Translate a give spin configuration bz one position along an axis."
        configuration = configuration.reshape(self.lattice_size)
        configuration = jnp.roll(configuration, 1, axis=self.axis).reshape(-1)
        return configuration


