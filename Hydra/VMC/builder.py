from copy import deepcopy

from netket.operator import AbstractOperator
from netket.utils.types import Optimizer, ScalarOrSchedule
from netket.vqs import MCState

from .drivers import REGISTRY_VMC


class VMCBuilder:

    def __init__(self, setup):
        self.selection = setup["selection"]
        self.setup = deepcopy(setup[self.selection])

    def build(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        variational_state: MCState,
        diag_shift: ScalarOrSchedule = 1e-4,
    ):

        driver_cls = REGISTRY_VMC[self.selection]

        return driver_cls(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            variational_state=variational_state,
            setup=self.setup,
            diag_shift=diag_shift,
        )
