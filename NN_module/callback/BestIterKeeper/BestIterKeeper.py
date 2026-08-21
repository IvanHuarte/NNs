import copy
from pathlib import Path

import flax.serialization
import numpy as np
import numpy.typing as npt


class BestIterKeeper:
    """Store the values of a bunch of quantities from the best iteration.

    "Best" is defined in the sense of lowest energy.

    Args:
        Hamiltonian: An array containing the Hamiltonian matrix.
        N: The number of spins in the chain.
        baseline: A lower bound for the V score. If the V score of the best
            iteration falls under this threshold, the process will be stopped
            early.
        filename: Either None or a file to write the best state to.
    """

    def __init__(
        self,
        epochs: int,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float = 1e-8,
        filename: Path | None = None,
        mode: str = "best_energy",
        balanced_setup: dict = {
            "start_stats": 2 / 10,
            "stats_window": 1 / 10,
            "normalizer": "Zscore",
            "selector": "linear",
        },
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf

        self.best_state_energy = np.inf
        self.best_state_vscore = np.inf
        self.best_step = 0
        self.best_state = None
        self.step_threshold = epochs // 10
        self.step = -1

        if mode == "best_energy":
            self.update = self.best_energy_update
        elif mode == "best_vscore":
            self.update = self.best_vscore_update
        elif mode == "always":
            self.update = self.always_update

        self.exit_msg = "Simulation finished with success"

    def best_energy_update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        self.step += 1

        vstate = driver.state

        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(log_data[driver._loss_name].variance)
        mean = np.real(log_data[driver._loss_name].mean)
        vscore_step = self.N * var / mean**2

        if self.step > self.step_threshold or self.best_state is None:
            if (
                self.best_state_energy > energy_step and vscore_step < 0.05
            ) or self.best_state is None:
                self.best_state = copy.copy(vstate)
                self.best_state_energy = energy_step
                self.best_state_vscore = vscore_step
                self.best_step = self.step

                if self.filename != None:
                    with open(self.filename, "wb") as file:
                        file.write(flax.serialization.to_bytes(driver.state))

        return self.survive_condition(energy_step, vscore_step)

    def best_vscore_update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        self.step += 1

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(log_data[driver._loss_name].variance)
        mean = np.real(log_data[driver._loss_name].mean)
        vscore_step = self.N * var / mean**2

        if self.step > self.step_threshold or self.best_state is None:

            if self.best_state_vscore > vscore_step or self.best_state is None:
                self.best_state = copy.copy(driver.state)
                self.best_state_energy = energy_step
                self.best_state_vscore = vscore_step
                self.best_step = self.step

                if self.filename != None:
                    with open(self.filename, "wb") as file:
                        file.write(flax.serialization.to_bytes(driver.state))

        return self.survive_condition(energy_step, vscore_step)

    def always_update(self, step, log_data, driver):

        self.step += 1

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(log_data[driver._loss_name].variance)
        mean = np.real(log_data[driver._loss_name].mean)
        vscore_step = self.N * var / mean**2

        # Always update
        self.best_state = copy.copy(driver.state)
        self.best_state_energy = energy_step
        self.best_state_vscore = vscore_step
        self.best_step = self.step

        if self.filename != None:
            with open(self.filename, "wb") as file:
                file.write(flax.serialization.to_bytes(driver.state))

        return self.survive_condition(energy_step, vscore_step)

    def balanced_update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(log_data[driver._loss_name].variance)
        mean = np.real(log_data[driver._loss_name].mean)
        vscore_step = self.N * var / mean**2

        # print(f"\nStep {step}:")
        # print(f"Energy: {energy_step:.6e}, Vscore: {vscore_step:.6e}")
        if self.step > self.start_stats:

            if self.step > self.start_stats + self.stats_window:
                norm_E, norm_V = self.normalizer(energy_step, vscore_step)
                score = self.selector(norm_E, norm_V)
                # print(f"norm_E = {norm_E:.6e}, norm_V = {norm_V:.6e}")
                # print(f"Score: {score:.6f}, Best score: {self.best_score:.6f}")

                if score < self.best_score:
                    # print(f"New best score found: {score:.6f} < {self.best_score:.6f}. Updating best state.")
                    self.best_score = score
                    self.best_state = copy.copy(driver.state)
                    self.best_state_energy = energy_step
                    self.best_state_vscore = vscore_step
                    self.best_step = self.step

                    if self.filename != None:
                        with open(self.filename, "wb") as file:
                            file.write(flax.serialization.to_bytes(driver.state))

            self.normalizer.update_history(energy_step, vscore_step)
            # print(f"Updating historial. Length: {len(self.normalizer.energy_history)}")

        return self.survive_condition(energy_step, vscore_step)

    def survive_condition(self, energy, vscore):

        survive = True
        if vscore < self.baseline:
            survive = False
            self.exit_msg = f"Vscore {vscore:.3e} is below baseline {self.baseline}"
            print(self.exit_msg)
        if not np.isfinite(energy):
            survive = False
            self.exit_msg = f"Energy has diverged ({energy:.3e}). Simulation crashed."
            print(self.exit_msg)

        return survive
