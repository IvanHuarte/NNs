#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python
"""
VMC simulation driven by 4 JSON files (order matters):
    config0 (simulation), config1 (coupling model), config2 (Hydra stages),
    config3 (Hydra module storage).

For each size in config1 'sizes' and each point of the config1 '*_list'
sweep: exact diagonalization (small systems), network + vstate, training
over the schedule periods (with architecture changes between stages), and
results of the best state kept by BestIterKeeper.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import copy
import json
import time
from dataclasses import dataclass

import jax
import netket as nk
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_debug_nans", True)

from CouplingModel import ModelFactory
from NN_module.callback import BestIterKeeper, Callback, Checkpoint
from NN_module.callback.utils import dump_callback
from NN_module.display import display_simulation_settings
from NN_module.exact_diagonalization import calc_exact_diag
from NN_module.label_utils import make_labels, make_run_paths
from NN_module.NN.Hydra import Hydra
from NN_module.NN.utils import make_setup_serializable
from NN_module.observables import measureNdump, phase_stats_vstate
from NN_module.sampler.sampler import SamplerFactory, resolve_sampler_setup
from NN_module.saveNload import save_results
from NN_module.schedule.schedule import Schedule
from NN_module.utils import print_tree, save_config_files
from NN_module.VMC import VMCBuilder

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
        model=model,
        seed=key_model,
        sampler_seed=key_sampler,
        n_samples=n_samples,
        n_discard_per_chain=0,
        chunk_size=sampler_setup["chunk_vstate"],
    )


def build_callbacks(cfg, sim_config, H, N, ed, labels, paths, total_epochs):
    callback_setup = cfg.sim["callback"]
    checkpoint = callback_setup["checkpoint"]
    callback_objects, callback_funcs = Callback(
        callback_setup,
        sim_config=sim_config,
        total_epochs=total_epochs,
        H=H,
        N=N,
        E_ED=ed.E_global,
        x_ED=ed.x_ref,
        sim_label_folder=paths.write_folder + labels.sim if checkpoint else "",
    )

    keeper = next((c for c in callback_objects if isinstance(c, BestIterKeeper)), None)
    if keeper is None:
        raise RuntimeError(
            "The results are taken from the best state: enable "
            "config0 'callback.keeper'."
        )
    checkpointer = next((c for c in callback_objects if isinstance(c, Checkpoint)), None)
    return keeper, checkpointer, callback_funcs


def print_period(i, total_periods, info, ds_info):
    epochs, mode = info[0][0], info[0][1]
    lr_string = "".join(f"{inf[2]}  " for inf in info)
    rescaled = "" if len(info) < 4 else info[3]
    print(f"\nPeriod {i + 1} / {total_periods}:")
    print(f"Training {mode} for {epochs} epochs")
    print(f"LR: {lr_string}  ({rescaled})")
    print(f"Diagonal shift: {ds_info}\n")


def train(cfg, hydra, vstate, H, schedule, ds_schedule, ds_info, callbacks, log, make_state):
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
        vmc.run(n_iter=info[0][0], out=log, callback=callbacks, show_progress=True)

        mean, std, psi = phase_stats_vstate(vstate)
        print(f"VS phase: {mean} ± {std}  ({psi})")

        if change:
            print("Changing Architecture")
            schedule.eon += 1
            hydra.arch_evol(vstate.parameters)  # n_stage + 1
            vstate = make_state(hydra.model, eon=schedule.eon)
            vstate.parameters = hydra.weight_transplantation(vstate.parameters)
            schedule.update_eon_code2path(hydra.get_code2path(vstate.parameters))

    return vstate, vmc_builder, time.time() - time_in


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

    labels = make_labels(cm_setup, cfg.nn_name, paths.uuid)

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
    keeper, checkpointer, callbacks = build_callbacks(
        cfg, sim_config, H, N, ed, labels, paths, schedule.total_epochs
    )
    log = nk.logging.RuntimeLog()
    vstate, vmc_builder, time_exe = train(
        cfg, hydra, vstate, H, schedule, ds_schedule, ds_info, callbacks, log, make_state
    )

    #### RESULTS OF THE BEST STATE ####
    vstate = keeper.best_state
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


def main():
    print("Ranks:", jax.process_count())
    print(f"Devices:{jax.devices()} ({jax.device_count()})")
    print("Local devices:", jax.local_devices())

    args = parse_args()
    cfg = load_configs(args.config or DEFAULT_CONFIGS)

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
