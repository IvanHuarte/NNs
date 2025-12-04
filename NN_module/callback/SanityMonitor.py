import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

from NN_module.callback.metrics import calc_metrics
from NN_module.callback.diagnose import diagnose_metrics


class SanityMonitor:

    def __init__(self, sanity_setup):

        self.do_each = sanity_setup["do_each"]

        self.metrics_setup = sanity_setup["metrics"]
        self.diagnose_setup = sanity_setup["diagnose"]

    def metrics(self, hamiltonian, vstate):

        grad_metrics = calc_metrics(hamiltonian, vstate, self.metrics_setup)

        return grad_metrics

    def diagnose(self, metrics):

        diagnosis = diagnose_metrics(metrics)
        return diagnosis

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

            print("=== SANITY CHECK ===")
            print(f"Step {step}")
            print(f"Metrics: \n{metrics}\n")
            print(f"Diagnosis: \n{diagnosis}\n")

        return True
