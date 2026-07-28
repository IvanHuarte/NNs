import netket as nk
from netket.operator import AbstractOperator
from netket.optimizer import SR
from netket.utils.types import Optimizer, ScalarOrSchedule
from netket.vqs import MCState

from ...utils import solver_dict


class VMC_netket(nk.driver.VMC):

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        variational_state: MCState,
        setup: dict,
        diag_shift: ScalarOrSchedule = 1e-4,
    ):

        preconditioner = self.process_setup(setup, diag_shift)

        super().__init__(
            hamiltonian,
            optimizer,
            variational_state=variational_state,
            **preconditioner,
        )

    def process_setup(self, setup, diag_shift):

        preconditioner = {}
        sr = setup["SR"]
        setup.pop("SR")

        # Set default if argument is None
        setup = dict([(k, v) for k, v in setup.items() if v not in [None, False]])

        for k, v in setup.items():
            if k == "solver":
                setup[k] = solver_dict[v]

        if sr:
            preconditioner["preconditioner"] = SR(diag_shift=diag_shift, **setup)

        return preconditioner
