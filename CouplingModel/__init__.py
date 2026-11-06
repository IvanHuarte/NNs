"""
Coupling models (spin Hamiltonians on lattices), integrated from
ATMOS_VA/VA/src/VA_project (lattice, model, initialize_model).

    ModelFactory(size, config1)  ->  iterate params, get_model()
    model.cm.build_hamiltonian(hilbert, S_operators=...)  ->  NetKet operator

The classical variational engine of VA_project (engine/, measurements/)
stays in ATMOS_VA.
"""

from .initialize_model import ModelFactory
from .model import REGISTRED_MODELS

__all__ = ["ModelFactory", "REGISTRED_MODELS"]
