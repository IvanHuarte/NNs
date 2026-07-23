import orbax.checkpoint as ocp
import numpy as np
from pathlib import Path


class Checkpoint:

    def __init__(self, H, sim_label_folder, setup):

        self.Hamiltonian = H
        self.do_each = setup["do_each"]
        self.asyn = setup["async"]
        self.step = -1

        self.checkpoint_path = sim_label_folder + "_checkpoint.orbax"

        manger_options = ocp.CheckpointManagerOptions(
            max_to_keep=1000000,
            save_interval_steps=1,
            create=True,
            cleanup_tmp_directories=True,
        )

        self.manager = ocp.CheckpointManager(
            self.checkpoint_path,
            item_names=("parameters", "sampler_state", "metrics"),
            options=manger_options,
        )

    def __call__(self, _, log_data, driver):

        self.step += 1

        if self.step % self.do_each != 0:
            return True

        print(f"\n CHECKPOINT!!! --> step {self.step}\n")

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = vstate.hilbert.size * var / mean**2

        metrics = {"energy": float(energy_step), "vscore": float(vscore_step)}

        self.manager.save(
            self.step,
            args=ocp.args.Composite(
                parameters=ocp.args.StandardSave(vstate.parameters),
                sampler_state=ocp.args.StandardSave(vstate.sampler_state),
                metrics=ocp.args.JsonSave(metrics),
            ),
        )

        if not self.asyn:
            self.manager.wait_until_finished()

        return True

    def wait(self):
        self.manager.wait_until_finished()
