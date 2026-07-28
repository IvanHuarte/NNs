import netket as nk
from netket.operator import AbstractOperator
from netket.utils.types import Optimizer, ScalarOrSchedule
from netket.vqs import MCState

from ...utils import solver_dict


class VMC_SR_netket(nk.driver.VMC_SR):

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        variational_state: MCState,
        setup: dict,
        diag_shift: ScalarOrSchedule = 1e-4,
        momentum: ScalarOrSchedule | None = None,
    ):

        cleaned_setup = self.process_setup(setup)

        super().__init__(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            diag_shift=diag_shift,
            momentum=momentum,
            variational_state=variational_state,
            **cleaned_setup,
        )

    def process_setup(self, setup):

        # Set default if argument is None
        setup = dict([(k, v) for k, v in setup.items() if v is not None])

        for k, v in setup.items():
            if k == "linear_solver":
                setup[k] = solver_dict[v]

        return setup
