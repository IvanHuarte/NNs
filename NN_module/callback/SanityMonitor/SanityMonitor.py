import numpy as np

from .diagnose import diagnose_metrics
from .display_diagnosis import (
    display_diagnosis_sanity_monitor,
    display_diagnosis_simple,
)
from .metrics import calc_metrics


class SanityMonitor:

    def __init__(self, sanity_setup):

        self.do_each = sanity_setup["do_each"]
        self.verbose = sanity_setup["verbose"]
        self.mode = sanity_setup["mode"]

        self.metrics_setup = sanity_setup["metrics"]

    def metrics(self, hamiltonian, vstate):

        grad_metrics = calc_metrics(hamiltonian, vstate, self.metrics_setup)
        return grad_metrics

    def diagnose(self, metrics):

        diagnosis = diagnose_metrics(metrics)
        return diagnosis

    def display(self, diagnosis):

        if self.mode == "rich":
            display_diagnosis_sanity_monitor(diagnosis, verbose=self.verbose)
        elif self.mode == "simple":
            display_diagnosis_simple(diagnosis, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown display mode '{self.mode}'")

    def __call__(self, step, log_data, driver):
        """Monitor the sanity of the training by printing relevant metrics.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        if step == 0:
            return True
        if step % self.do_each == 0:

            vstate = driver.state
            N = vstate.hilbert.size

            energy_step = np.real(vstate.expect(driver._ham).mean)
            var = np.real(getattr(log_data[driver._loss_name], "variance"))
            mean = np.real(getattr(log_data[driver._loss_name], "mean"))
            vscore_step = N * var / mean**2

            metrics = self.metrics(driver._ham, vstate)
            diagnosis = self.diagnose(metrics)

            self.display(diagnosis)

        return True
