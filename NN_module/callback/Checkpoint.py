import orbax.checkpoint as ocp
import numpy as np


class Checkpoint:

    def __init__(self, H, sim_label_folder, setup):

        self.Hamiltonian = H

        checkpoint_path = sim_label_folder + "_checkpoint.orbax"
        print(checkpoint_path)
        manger_options = ocp.CheckpointManagerOptions(**setup["options"])
        self.do_each = setup["do_each"]
        if setup["async"]:
            checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
        else:
            checkpointer = ocp.PyTreeCheckpointHandler()

        self.manager = ocp.CheckpointManager(
            checkpoint_path, checkpointer, manger_options
        )

    def __call__(self, step, log_data, driver):

        if step % self.do_each != 0 or step == 0:
            return True

        vstate = driver.state
        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = vstate.hilbert.size * var / mean**2

        items = {"parameters": vstate.parameters, "sampler_state": vstate.sampler_state}
        metrics = {"energy": energy_step, "vscore": vscore_step}

        self.manager.save(step=step, items=items, metrics=metrics)

        return True
