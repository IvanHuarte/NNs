"""
Periodic checkpoints of the full variational state (parameters + sampler
state + metrics) with orbax:

    <sim>_checkpoint.orbax/<global step>/{parameters, sampler_state, metrics}

config0 "checkpoint_setup":
    do_each       steps between checkpoints (default 500)
    async         true: the training does not wait for the disk (default true)
    max_to_keep   keep only the newest n checkpoints (default null = all)
    keep_every    with max_to_keep, also keep every checkpoint whose step is
                  a multiple of this (default null)
    stage_end     also checkpoint the last step of every Hydra stage (default true)

Asynchronous mode
-----------------
The device -> host copy is done in the callback (jax.device_get, a few ms
for these networks); orbax then serialises and writes in a background
thread while the training goes on. Only one checkpoint is in flight: a new
save waits for the previous one. `flush()` (end of a stage, end of the run
and on a crash, called by VMC_simulation.train) waits for the pending write,
so the last checkpoint is always complete on disk. Previously the manager
was never waited for at the end of the run, so the last asynchronous write
was lost (or failed with "cannot schedule new futures after interpreter
shutdown").

Metrics (JSON): energy, energy_error, variance, vscore, stage, local_step.
Loading: config2 load_model {"checkpoint": <step>, "load_from": ["…_checkpoint.orbax"]}.
"""

from __future__ import annotations

import os
import shutil

import jax
import numpy as np
import orbax.checkpoint as ocp


class Checkpoint:
    def __init__(self, H, sim_label_folder, setup, stage_epochs=None):
        self.do_each = int(setup.get("do_each", 500))
        self.asyn = bool(setup.get("async", True))
        self.stage_end = bool(setup.get("stage_end", True))
        self.stage_epochs = list(stage_epochs or [])
        self.step = -1
        self.stage = 0
        self.stage_start = 0
        self.saved = []

        self.checkpoint_path = os.path.abspath(sim_label_folder + "_checkpoint.orbax")
        if os.path.isdir(self.checkpoint_path):
            shutil.rmtree(self.checkpoint_path)  # stale run with the same label

        max_to_keep = setup.get("max_to_keep")
        keep_every = setup.get("keep_every")
        options = ocp.CheckpointManagerOptions(
            max_to_keep=int(max_to_keep) if max_to_keep else None,
            keep_period=int(keep_every) if keep_every else None,
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=self.asyn,
        )
        self.manager = ocp.CheckpointManager(
            self.checkpoint_path,
            item_names=("parameters", "sampler_state", "metrics"),
            options=options,
        )

    # ------------------------------------------------------------------

    def _is_stage_end(self):
        if not self.stage_end or not self.stage_epochs:
            return False
        k = min(self.stage, len(self.stage_epochs) - 1)
        return self.step - self.stage_start == self.stage_epochs[k] - 1

    def __call__(self, _, log_data, driver):
        self.step += 1
        if self.step % self.do_each != 0 and not self._is_stage_end():
            return True
        self.save(log_data, driver)
        return True

    def save(self, log_data, driver):
        stats = log_data[driver._loss_name]
        mean = float(np.real(stats.mean))
        var = float(np.real(stats.variance))
        metrics = {
            "energy": mean,
            "energy_error": float(np.real(stats.error_of_mean)),
            "variance": var,
            "vscore": driver.state.hilbert.size * var / mean**2 if mean else None,
            "stage": self.stage,
            "local_step": self.step - self.stage_start,
        }
        # host copy now: the background write never touches device buffers
        params = jax.device_get(driver.state.parameters)
        sampler_state = jax.device_get(driver.state.sampler_state)

        self.manager.save(
            self.step,
            args=ocp.args.Composite(
                parameters=ocp.args.StandardSave(params),
                sampler_state=ocp.args.StandardSave(sampler_state),
                metrics=ocp.args.JsonSave(metrics),
            ),
            force=True,
        )
        self.saved.append(self.step)
        if not self.asyn:
            self.manager.wait_until_finished()

    # hooks used by VMC_simulation.train -------------------------------

    def flush(self, wait=True):
        if wait:
            self.manager.wait_until_finished()

    def new_stage(self, stage=None):
        self.flush()
        self.stage = self.stage + 1 if stage is None else stage
        self.stage_start = self.step + 1

    def finalize(self):
        self.flush()

    def wait(self):  # old name
        self.flush()

    def close(self):
        self.manager.wait_until_finished()
        self.manager.close()
