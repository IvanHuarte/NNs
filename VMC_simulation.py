#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python
"""
VMC simulation driven by 4 JSON files (order matters):
    config0 (simulation), config1 (coupling model), config2 (Hydra stages),
    config3 (Hydra module storage).

For each size in config1 'sizes' and each point of the config1 '*_list'
sweep: exact diagonalization (small systems), network + vstate, training
over the schedule periods (with architecture changes between stages), and
results of the best state kept by BestIterKeeper (on disk with orbax, one
best per mode and stage; the results use the primary mode, last stage).
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import copy
import json
import signal
import time
from dataclasses import dataclass

import jax
import netket as nk
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_debug_nans", True)

from CouplingModel import ModelFactory
from Hydra.callback import BestIterKeeper, Callback, Checkpoint
from Hydra.callback.utils import dump_callback
from Hydra.display import display_simulation_settings
from Hydra.exact_diagonalization import calc_exact_diag, irrep_exact_diag, stage_sectors
from Hydra.label_utils import arch_label, make_labels, make_run_paths
from Hydra.NN.Hydra import Hydra
from Hydra.NN.precision import set_matmul_precision, set_network_dtype, state_kwargs
from Hydra.NN.utils import make_setup_serializable
from Hydra.observables import measureNdump, phase_stats_vstate
from Hydra.sampler.sampler import SamplerFactory, resolve_sampler_setup
from Hydra.saveNload import save_results
from Hydra.schedule.schedule import Schedule
from Hydra.utils import print_tree, save_config_files
from Hydra.VMC import VMCBuilder

HOME = os.path.expanduser("~")
DEFAULT_CONFIGS = [
    HOME + "/Escritorio/Ivan/NNs/config0.json",
    HOME + "/Escritorio/Ivan/NNs/config1_CM.json",
    HOME + "/Escritorio/Ivan/NNs/config2_Hydra.json",
    HOME + "/Escritorio/Ivan/NNs/config3_Hydra_NN.json",
]


# ============================================================
# Configuration
# ============================================================


@dataclass
class Configs:
    paths: list
    sim: dict  # config0
    cm: dict  # config1
    hydra: dict  # config2
    hydra_nn: dict  # config3

    @property
    def nn(self):
        """config2 + config3 (what Hydra receives)."""
        return {**self.hydra, **self.hydra_nn}

    @property
    def nn_name(self):
        return self.hydra["selection"]

    @property
    def nn_evol_setup(self):
        return self.nn[self.nn_name]

    @property
    def as_list(self):
        return [self.sim, self.cm, self.hydra, self.hydra_nn]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        required=False,
        help="Parse configuration files in order. Simulation/CM/Hydra/Hydra_NN",
    )
    return parser.parse_args()


def load_configs(paths):
    print("Configurations:")
    for c in paths:
        print(f" - {c}")
    print()

    dicts = []
    for path in paths:
        with open(path, "r") as f:
            dicts.append(json.load(f))
    return Configs(paths, *dicts)


def get_seed(config):
    """config0 'seed' (int) for reproducible runs; time-based if absent/null."""
    seed = config.get("seed")
    return int(time.time()) if seed is None else int(seed)


# ============================================================
# Building blocks
# ============================================================


def make_vstate(sampler, model, sampler_setup, n_samples, key):
    """MCState with explicit seeds for the parameters and the sampler."""
    key_model, key_sampler = jax.random.split(key)
    return nk.vqs.MCState(
        sampler=sampler,
        seed=key_model,
        sampler_seed=key_sampler,
        n_samples=n_samples,
        n_discard_per_chain=0,
        chunk_size=sampler_setup["chunk_vstate"],
        **state_kwargs(model),  # model, or mixed precision (config0 network_dtype)
    )


def build_callbacks(cfg, sim_config, H, N, ed, labels, paths, total_epochs, stage_epochs):
    callback_setup = cfg.sim["callback"]
    checkpoint = callback_setup["checkpoint"]
    callback_objects, callback_funcs = Callback(
        callback_setup,
        sim_config=sim_config,
        total_epochs=total_epochs,
        H=H,
        N=N,
        E_ED=ed.E_global,
        E_ref=ed.E_ref,
        x_ED=ed.x_ref,
        sim_label_folder=paths.write_folder + labels.sim if checkpoint else "",
        stage_epochs=stage_epochs,
        best_label_folder=paths.write_folder + labels.sim,
    )

    keeper = next((c for c in callback_objects if isinstance(c, BestIterKeeper)), None)
    if keeper is None:
        raise RuntimeError(
            "The results are taken from the best state: enable "
            "config0 'callback.keeper'."
        )
    checkpointer = next((c for c in callback_objects if isinstance(c, Checkpoint)), None)
    return keeper, checkpointer, callback_objects, callback_funcs


def _hook(objects, name, *args, **kwargs):
    """Call objects[i].<name>(...) on every callback that implements it."""
    for obj in objects:
        fn = getattr(obj, name, None)
        if callable(fn):
            fn(*args, **kwargs)


def print_period(i, total_periods, info, ds_info):
    epochs, mode = info[0][0], info[0][1]
    lr_string = "".join(f"{inf[2]}  " for inf in info)
    rescaled = "" if len(info) < 4 else info[3]
    print(f"\nPeriod {i + 1} / {total_periods}:")
    print(f"Training {mode} for {epochs} epochs")
    print(f"LR: {lr_string}  ({rescaled})")
    print(f"Diagonal shift: {ds_info}\n")


def train(cfg, hydra, vstate, H, schedule, ds_schedule, ds_info, callbacks, log, make_state,
          callback_objects):
    """
    Run every schedule period. At a stage change the architecture evolves
    (Hydra.arch_evol) and a new vstate receives the transplanted weights.
    Returns the last vstate, the last VMC builder and the execution time.
    """
    H_jax = H.to_jax_operator()
    time_in = time.time()

    for i, (lr_period, info, change) in enumerate(schedule.schedule()):
        print_period(i, schedule.total_periods, info, ds_info[i])

        optimizer = schedule.transform_optimizer(
            vstate.parameters, cfg.sim["optimizer"], info, lr_period
        )
        vmc_builder = VMCBuilder(cfg.sim["VMC"])
        vmc = vmc_builder.build(
            hamiltonian=H_jax,
            optimizer=optimizer,
            variational_state=vstate,
            diag_shift=ds_schedule[i],
        )
        print(f"\nVMC selection: {vmc_builder.selection}")
        print(f"\nTraining {info[0][1]} for {info[0][0]} epochs...")
        try:
            vmc.run(n_iter=info[0][0], out=log, callback=callbacks, show_progress=True)
        except BaseException:
            # crash (OOM, Ctrl-C, scancel via SIGTERM): finish the pending writes
            # (keeper bests, checkpoints) before the interpreter shuts the orbax
            # threads down
            _hook(callback_objects, "flush", wait=True)
            raise

        mean, std, psi = phase_stats_vstate(vstate)
        print(f"VS phase: {mean} ± {std}  ({psi})")

        if change:
            print("Changing Architecture")
            schedule.eon += 1
            # final state, or the stage best if the stage ended unstable
            keeper = next((c for c in callback_objects if hasattr(c, "stage_end_params")), None)
            params = keeper.stage_end_params(vstate) if keeper else vstate.parameters
            hydra.arch_evol(params)  # n_stage + 1
            vstate = make_state(hydra.model, eon=schedule.eon)
            vstate.parameters = hydra.weight_transplantation(vstate.parameters)
            schedule.update_eon_code2path(hydra.get_code2path(vstate.parameters))
            _hook(callback_objects, "new_stage", schedule.eon)

    _hook(callback_objects, "finalize")
    return vstate, vmc_builder, time.time() - time_in


def stage_ed_errors(summary, config_nn, hi, H, size, ed):
    """
    Add the ED energy of its sector and the relative error to the best of
    each mode and stage (small systems only; the last stage uses `ed`).
    """
    if not ed.available:
        return summary
    sectors = stage_sectors(config_nn)
    cache = {}
    for entries in summary.values():
        for e in entries:
            symmetries, irrep = sectors[e["stage"]]
            if not irrep:
                E0 = ed.E_global
            elif (symmetries, irrep) == (ed.symmetries, ed.irrep):
                E0 = ed.E_irrep
            else:
                if (symmetries, irrep) not in cache:
                    cache[(symmetries, irrep)] = float(
                        irrep_exact_diag(hi, H, symmetries, irrep, size)[0]
                    )
                E0 = cache[(symmetries, irrep)]
            e["sector"] = {"symmetries": list(symmetries), "irrep": list(irrep)}
            e["E_ED"] = E0
            e["error"] = abs(e["energy"] - E0) / abs(E0) if E0 else None
    return summary


# ============================================================
# One simulation point
# ============================================================


def run_point(cfg, size, hi, paths, model_factory, sampler_setup, n_samples, n_ranks):
    N = int(np.prod(size))
    cm_setup = model_factory.get_setup()
    cm_model = model_factory.get_model()  # completes cm_setup params (Oxalate)

    # Hamiltonian in the units chosen in config1 (Pauli or S = sigma/2)
    H = cm_model.cm.build_hamiltonian(hi, S_operators=cfg.cm["S_operators"])

    ed = calc_exact_diag(hilbert=hi, hamiltonian=H, config_nn=cfg.nn, lattice_size=size)
    ed.report()

    arch = arch_label(cfg.nn_evol_setup)
    if cfg.sim.get("network_dtype", "float64") != "float64":
        arch += "_" + cfg.sim["network_dtype"].replace("float", "f")
    labels = make_labels(cm_setup, cfg.nn_name, paths.uuid, arch=arch)

    #### NETWORK, SAMPLER, VSTATE ####
    hydra = Hydra(cfg.nn, lattice_size=size, hilbert=hi)
    print(f"Total samples: {n_samples}\n")
    print_tree(hydra.setup, values=True)

    print("Initializing sampler...")
    sampler = SamplerFactory(sampler_setup, cm_model=cm_model).get_sampler(hi)

    print("Initializing Variational State...")
    seed = get_seed(cfg.sim)
    key = jax.random.key(seed)

    def make_state(model, eon=0):
        return make_vstate(
            sampler, model, sampler_setup, n_samples, jax.random.fold_in(key, eon)
        )

    vstate = make_state(hydra.model)
    print("Variational state initialized.")
    if hydra.load_model:  # transplant loaded parameters
        vstate = hydra.load_vstate(vstate)

    #### SCHEDULE ####
    schedule = Schedule(cfg.sim["schedule"], hydra.get_code2path(vstate.parameters))
    ds_schedule, ds_info = schedule.get_diag_shift_schedule()

    sim_config = {
        "SIM": copy.deepcopy(cfg.sim),
        "CM": cm_setup,
        "NN": {"name": cfg.nn_name, "setup": cfg.nn_evol_setup},
    }
    sim_config["SIM"]["schedule"]["total_epochs"] = schedule.total_epochs

    display_simulation_settings(
        cm_setup=cm_setup,
        nn_name=cfg.nn_name,
        nn_evol_setup=cfg.nn_evol_setup,
        sampler_setup=sampler_setup,
        n_samples=n_samples,
        n_ranks=n_ranks,
        schedule=schedule,
        ds_info=ds_info,
        optimizer_setup=cfg.sim["optimizer"],
        vmc_setup=cfg.sim["VMC"],
    )

    #### TRAINING ####
    keeper, checkpointer, callback_objects, callbacks = build_callbacks(
        cfg, sim_config, H, N, ed, labels, paths, schedule.total_epochs,
        schedule.stage_epochs,
    )
    log = nk.logging.RuntimeLog()
    vstate, vmc_builder, time_exe = train(
        cfg, hydra, vstate, H, schedule, ds_schedule, ds_info, callbacks, log,
        make_state, callback_objects,
    )

    #### RESULTS OF THE BEST STATE (primary mode, last stage) ####
    vstate = keeper.restore(vstate)
    eon, _, _ = schedule.locate_period(keeper.best_step)
    sim_config["NN"] = {
        "name": cfg.nn_name + f"_stage{eon}",
        "setup": hydra.storage[f"stage{eon}"],
    }

    schedule_flat = schedule.flat_setup()
    do_each_checkpoint = (
        cfg.sim["callback"]["checkpoint_setup"]["do_each"] if checkpointer else None
    )
    callback_artifacts = dump_callback(
        log,
        ed=ed,
        labels=labels,
        write_folder=paths.write_folder,
        size=size,
        time_exe=time_exe,
        best_step=keeper.best_step,
        best_energy=keeper.best_state_energy,
        schedule_setup=schedule_flat,
        do_each_checkpoint=do_each_checkpoint,
        message=keeper.exit_msg,
    )

    results, mp_array_vs, mp_array_ED_global, mp_array_ED_irrep = measureNdump(
        keeper, time_exe, ed, S_operators=cfg.cm["S_operators"]
    )
    results["key"] = jax.random.key_data(key).tolist()
    results["seed"] = seed
    results["mssg"] = keeper.exit_msg
    results["schedule_flat"] = schedule_flat
    results["keeper"] = {
        "primary": keeper.primary,
        "bests": stage_ed_errors(keeper.summary(), cfg.nn, hi, H, size, ed),
    }

    sim_config["SIM"]["sampler"]["nsamples"] = n_samples
    sim_config["SIM"]["sampler"]["final_rng"] = jax.random.key_data(
        vstate.sampler_state.rng
    ).tolist()
    sim_config["SIM"]["VMC"] = {
        "selection": vmc_builder.selection,
        "setup": vmc_builder.setup,
    }

    dump_setup = {
        **sim_config,
        "results": results,
        "_artifacts": {"callback": callback_artifacts},
        "labels": labels.as_dict(),
    }
    if checkpointer is not None:
        dump_setup["_artifacts"]["checkpoint"] = checkpointer.checkpoint_path
    if keeper.orbax_path is not None:
        dump_setup["_artifacts"]["best"] = keeper.orbax_path
    dump_setup = make_setup_serializable(dump_setup)

    # The UUID folder holds the whole sweep: its legend describes the sweep
    sweep = {
        "name": cm_setup["name"],
        "size": cm_setup["size"],
        "mode": cfg.cm["mode"],
        "sweep": cfg.cm["CM"][cm_setup["name"]],
        "kwargs_lattice": cm_setup["kwargs_lattice"],
        "S_operators": cm_setup["S_operators"],
    }
    legend = make_setup_serializable(
        {"NN": sim_config["NN"], "SIM": sim_config["SIM"], "CM": sweep}
    )

    save_results(
        vstate,
        dump_setup,
        x_ED_global=ed.x_global,
        x_ED_irrep=ed.x_irrep,
        modphase=mp_array_vs,
        modphase_ED_global=mp_array_ED_global,
        modphase_ED_irrep=mp_array_ED_irrep,
        write_folder=paths.write_folder,
        sim_label=labels.sim,
        ED_label=labels.ED,
        json_label=labels.json,
        sim_uuid=paths.uuid,
        legend=legend,
    )


# ============================================================
# Main
# ============================================================


def _sigterm_to_exit(signum, frame):
    """scancel/SIGTERM -> SystemExit, so train() flushes the keeper bests."""
    raise SystemExit(f"terminated by signal {signum}")


def main():
    signal.signal(signal.SIGTERM, _sigterm_to_exit)
    print("Ranks:", jax.process_count())
    print(f"Devices:{jax.devices()} ({jax.device_count()})")
    print("Local devices:", jax.local_devices())

    args = parse_args()
    cfg = load_configs(args.config or DEFAULT_CONFIGS)
    # float64 unless config0 "network_dtype" says otherwise (see NN.precision)
    set_network_dtype(cfg.sim.get("network_dtype", "float64"))
    set_matmul_precision(cfg.sim.get("matmul_precision", "highest"))

    sampler_setup = resolve_sampler_setup(cfg.sim["sampler"], verbose=True)
    n_ranks = jax.device_count()
    n_samples = (
        n_ranks
        * sampler_setup["n_samples_per_chain"]
        * sampler_setup["n_chains_per_rank"]
    )

    for size in cfg.cm["sizes"]:
        paths = make_run_paths(
            cfg.sim, cfg.cm["CM"]["selection"], cfg.nn_name, cfg.nn_evol_setup, size
        )
        save_config_files(paths.write_folder, cfg.as_list, cfg.paths, size)

        hi = nk.hilbert.Spin(s=1 / 2, N=int(np.prod(size)), total_sz=cfg.sim["sz_total"])
        model_factory = ModelFactory(size, cfg.cm)

        for _ in model_factory.get_params():
            run_point(
                cfg, size, hi, paths, model_factory, sampler_setup, n_samples, n_ranks
            )


if __name__ == "__main__":
    main()
