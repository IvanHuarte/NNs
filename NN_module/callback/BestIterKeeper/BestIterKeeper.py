"""
Best states of a VMC run, kept on disk with orbax instead of in VRAM.

Several criteria ("modes") can run at the same time; each one keeps its own
best state per stage (Hydra eon):

    best_energy   lowest energy among the states with vscore < vscore_max
                  (if no state has passed that bound yet, the lowest energy)
    best_vscore   lowest vscore
    best          lowest joint score  S = E + |E| * V / c   (see below)
    always        the last state (debug)

Joint score ("best")
--------------------
For a state close to an eigenstate E - E0 ~ Var / Delta: energy and variance
are two estimates of the same error. In relative terms the variance estimate
is eps_V = V / c, with V = N Var / E^2 the V-score (size intensive, model
independent). S = E + |E| eps_V = E + N Var / (c |E|) adds both evidences
in energy units, so

    - a small rise of E is accepted if V drops enough  (|E| dV / c > dE)
    - a big drop of E is accepted even if V rises a bit.

c was calibrated on J1J2 4x4/6x6 runs against ED (eps = (E - E0)/|E|):
V/eps = 5-16, median ~8 for converged states. Optional noise margin:
a candidate must beat the best score by z * sigma_dE, with
sigma_dE = sqrt((Var + Var_best) / Ns) (z = 0 by default).

Storage
-------
On every new best only a host (CPU RAM) copy of the parameters is taken
(jax.device_get); nothing extra stays in VRAM. Every `save_every` steps, at
each stage change and at the end, the pending bests are written with orbax
into ONE folder, one subfolder per mode:

    <sim>_best.orbax/
        best/          {parameters, sampler_state, metrics}
        best_energy/   {parameters, sampler_state, metrics}
        best_vscore/   ...

With several Hydra stages each mode keeps the best of every stage:

    <sim>_best.orbax/<mode>/stage<k>/{parameters, sampler_state, metrics}

`metrics/metadata` (JSON) holds energy, variance, vscore, score, step, stage. Each
(mode, stage) folder is overwritten in place by an orbax AsyncCheckpointer
(write to a temporary folder, then rename: a dead job leaves the previous
best intact). The selection is done here, not by orbax' best_fn, because
the criteria change along the run (vscore bound, noise margin).

Trace: with orbax_path, every step (step, stage, local_step, energy,
variance, vscore) is appended to <sim>_trace.txt at each write, so long
runs can be followed live and nothing is lost if the job dies.

After training, `restore(vstate, mode)` loads the best parameters of the
last stage into vstate and sets best_state / best_step / best_state_energy /
best_state_vscore (the API used by observables.measureNdump).
"""

import os
import shutil

import flax.serialization
import jax
import numpy as np
import numpy.typing as npt
import orbax.checkpoint as ocp

MODES = ("best_energy", "best_vscore", "best", "always")


class _Tracker:
    """Best state of one criterion inside the current stage."""

    def __init__(self, mode, setup):
        if mode not in MODES:
            raise ValueError(f"keeper mode {mode!r}, expected some of {MODES}")
        self.mode = mode
        self.vscore_max = setup["vscore_max"]
        self.c = setup["c"]
        self.z = setup["z"]
        self.best = None  # dict: energy, variance, vscore, score, step, stage, ...
        self.pending = None  # host copy waiting to be written
        self.saved_step = None
        self.manager = None
        self.history = []  # bests of finished stages

    def score(self, rec):
        return rec["energy"] + abs(rec["energy"]) * rec["vscore"] / self.c

    def is_better(self, rec):
        b = self.best
        if b is None or self.mode == "always":
            return True
        if self.mode == "best_vscore":
            return rec["vscore"] < b["vscore"]
        if self.mode == "best_energy":
            if self.vscore_max is not None:
                ok_new = rec["vscore"] < self.vscore_max
                ok_old = b["vscore"] < self.vscore_max
                if ok_new != ok_old:
                    return ok_new
            return rec["energy"] < b["energy"]
        # best: joint score with optional noise margin
        margin = 0.0
        if self.z:
            margin = self.z * np.sqrt((rec["variance"] + b["variance"]) / rec["n_samples"])
        return self.score(rec) < self.score(b) - margin


class BestIterKeeper:
    """
    Args:
        epochs: total epochs of the run.
        Hamiltonian, N: operator and number of sites (N enters the V-score).
        baseline: stop when the vscore falls under it.
        filename: legacy, flax bytes of the primary best state (None: off).
        mode: one mode or a list of them (the first is the primary one,
            used for the results). See MODES.
        orbax_path: folder of the on-disk bests (None: host RAM only).
        save_every: steps between writes of pending bests.
        start_fraction: within each stage, states are eligible after this
            fraction of the stage epochs (the first state is always taken).
        vscore_max: bound of best_energy (None: no bound).
        best_setup: {"c": 8.0, "z": 0.0} of the joint score.
        stage_epochs: epochs of each stage (default: one stage).
    """

    def __init__(
        self,
        epochs: int,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float = 1e-8,
        filename=None,
        mode="best_energy",
        orbax_path: str | None = None,
        save_every: int = 50,
        start_fraction: float = 0.1,
        vscore_max: float | None = 0.05,
        best_setup: dict | None = None,
        stage_epochs: list | None = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.save_every = max(1, int(save_every))
        self.start_fraction = start_fraction
        self.stage_epochs = list(stage_epochs) if stage_epochs else [epochs]
        self.orbax_path = (
            os.path.abspath(orbax_path.rstrip("/")) if orbax_path else None
        )
        if self.orbax_path and os.path.isdir(self.orbax_path):
            shutil.rmtree(self.orbax_path)  # stale bests of a previous run
        self.trace_path = None
        self._trace = []
        if self.orbax_path:
            base = self.orbax_path[: -len("_best.orbax")] if self.orbax_path.endswith("_best.orbax") else self.orbax_path
            self.trace_path = base + "_trace.txt"
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            with open(self.trace_path, "w") as f:
                f.write("# step stage local_step energy variance vscore\n")

        setup = {"vscore_max": vscore_max, "c": 8.0, "z": 0.0, **(best_setup or {})}
        modes = [mode] if isinstance(mode, str) else list(mode)
        if not modes:
            raise ValueError("keeper: at least one mode is needed")
        self.modes = list(dict.fromkeys(modes))
        self.primary = self.modes[0]
        self.trackers = {m: _Tracker(m, setup) for m in self.modes}

        self.step = -1
        self.stage = 0
        self.stage_start = 0
        self._host_cache = (None, None)  # (step, (params, sampler_state))

        # Filled by restore() (API of measureNdump)
        self.best_state = None
        self.best_state_energy = np.inf
        self.best_state_vscore = np.inf
        self.best_step = 0

        self.exit_msg = "Simulation finished with success"

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    @property
    def threshold(self):
        n = self.stage_epochs[min(self.stage, len(self.stage_epochs) - 1)]
        return int(self.start_fraction * n)

    def update(self, step, log_data, driver):
        """NetKet callback: (step, log_data, driver) -> keep running."""
        self.step += 1
        stats = log_data[driver._loss_name]
        energy = float(np.real(stats.mean))
        variance = float(np.real(stats.variance))
        vscore = self.N * variance / energy**2
        rec = {
            "energy": energy,
            "variance": variance,
            "vscore": vscore,
            "step": self.step,
            "local_step": self.step - self.stage_start,
            "stage": self.stage,
            "n_samples": int(driver.state.n_samples),
        }

        if self.trace_path:
            self._trace.append(
                f"{self.step} {self.stage} {rec['local_step']} {energy:.10g} {variance:.10g} {vscore:.10g}\n"
            )

        eligible = rec["local_step"] > self.threshold
        for tr in self.trackers.values():
            if (eligible or tr.best is None) and tr.is_better(rec):
                tr.best = rec
                tr.pending = self._host_copy(driver.state)
                if tr.mode == self.primary and self.filename is not None:
                    with open(self.filename, "wb") as file:
                        file.write(flax.serialization.to_bytes(driver.state))

        if self.orbax_path and self.step % self.save_every == 0:
            self.flush(wait=False)

        survive = self.survive_condition(energy, vscore)
        if not survive:
            self.flush(wait=True)
        return survive

    # Old name of the callback method
    __call__ = update

    def _host_copy(self, vstate):
        """Parameters + sampler state on host, shared by the modes of a step."""
        step, data = self._host_cache
        if step != self.step:
            data = (
                jax.device_get(vstate.parameters),
                jax.device_get(vstate.sampler_state),
            )
            self._host_cache = (self.step, data)
        return data

    # ------------------------------------------------------------------
    # Stages and disk
    # ------------------------------------------------------------------

    def _folder(self, mode, stage):
        """<orbax_path>/<mode>[/stage<k>] (stage level only with several stages)."""
        parts = [self.orbax_path, mode]
        if len(self.stage_epochs) > 1:
            parts.append(f"stage{stage}")
        return os.path.join(*parts)

    def _manager(self, tr):
        if tr.manager is None:
            tr.manager = ocp.AsyncCheckpointer(ocp.CompositeCheckpointHandler())
        return tr.manager

    def _metrics(self, tr):
        b = tr.best
        out = {k: b[k] for k in ("energy", "variance", "vscore", "step", "local_step", "stage")}
        out["mode"] = tr.mode
        out["score"] = tr.score(b)
        return out

    def flush(self, wait=True):
        """Write the pending bests (and the trace) to disk."""
        if not self.orbax_path:
            return
        if self._trace:
            with open(self.trace_path, "a") as f:
                f.writelines(self._trace)
            self._trace = []
        for tr in self.trackers.values():
            if tr.pending is None or tr.best["step"] == tr.saved_step:
                continue
            params, sampler_state = tr.pending
            manager = self._manager(tr)
            manager.wait_until_finished()  # one write at a time per mode
            manager.save(
                self._folder(tr.mode, tr.best["stage"]),
                args=ocp.args.Composite(
                    parameters=ocp.args.StandardSave(params),
                    sampler_state=ocp.args.StandardSave(sampler_state),
                    metrics=ocp.args.JsonSave(self._metrics(tr)),
                ),
                force=True,
            )
            tr.saved_step = tr.best["step"]
            tr.pending = None  # free host RAM: it is on disk now
        if wait:
            for tr in self.trackers.values():
                if tr.manager is not None:
                    tr.manager.wait_until_finished()

    def _close_stage(self):
        self.flush(wait=True)
        for tr in self.trackers.values():
            if tr.best is not None:
                entry = self._metrics(tr)
                entry["path"] = self.best_path(tr.mode, tr.best["stage"])
                if self.orbax_path is None:
                    entry["_params"] = tr.pending[0] if tr.pending else None
                tr.history.append(entry)
            if tr.manager is not None:
                tr.manager.close()
            tr.manager = None

    def new_stage(self, stage=None):
        """Call at a Hydra stage change (new architecture/irrep)."""
        self._close_stage()
        self.stage = self.stage + 1 if stage is None else stage
        self.stage_start = self.step + 1
        for tr in self.trackers.values():
            tr.best, tr.pending, tr.saved_step = None, None, None

    def finalize(self):
        """End of training: write everything, keep the bests of the last stage."""
        last = {m: tr.best for m, tr in self.trackers.items()}
        pending = {m: tr.pending for m, tr in self.trackers.items()}
        self._close_stage()
        for m, tr in self.trackers.items():  # restore() reads the last stage
            tr.best = last[m]
            tr.pending = pending[m] if self.orbax_path is None else None

    def best_path(self, mode, stage):
        """Folder of the best of (mode, stage) on disk, or None."""
        if not self.orbax_path:
            return None
        path = self._folder(mode, stage)
        return path if os.path.isdir(os.path.join(path, "parameters")) else None

    def summary(self):
        """{mode: [best of each stage]} for the results JSON."""
        return {
            m: [{k: v for k, v in e.items() if not k.startswith("_")} for e in tr.history]
            for m, tr in self.trackers.items()
        }

    def restore(self, vstate, mode=None):
        """
        Load into vstate the best parameters of `mode` (default: primary)
        in the last stage and expose them as best_state/best_* .
        """
        mode = mode or self.primary
        tr = self.trackers[mode]
        if tr.best is None:
            raise RuntimeError(f"keeper: no state was kept for mode {mode!r}")
        if self.orbax_path:
            path = self.best_path(mode, tr.best["stage"])
            params = ocp.PyTreeCheckpointer().restore(os.path.join(path, "parameters"))
        else:
            params = tr.pending[0]
        # same container types as the live parameters (dict/FrozenDict)
        vstate.parameters = flax.serialization.from_state_dict(vstate.parameters, params)
        self.best_state = vstate
        self.best_state_energy = tr.best["energy"]
        self.best_state_vscore = tr.best["vscore"]
        self.best_step = tr.best["step"]
        return vstate

    # ------------------------------------------------------------------

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
