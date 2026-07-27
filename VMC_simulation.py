#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python
import argparse
import json
import time
import uuid

import os
import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer import solver
import numpy as np
import optax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_debug_nans", True)

print("Ranks:", jax.process_count())
print("Devices:", jax.devices())
print("Devices:", jax.device_count())

# Añadir los directorios necesarios

from VA_project.engine.runners import Runner

# Importar módulos necesarios
from VA_project.initialize_model import ModelFactory

from NN_module.callback import Callback
from NN_module.callback.utils import dump_callback
from NN_module.label_utils import (
    display_simulation_settings,
    get_filenames_from_settings,
    get_write_folder_from_model,
)
from NN_module.NN.Hydra import Hydra
from NN_module.NN.utils import make_setup_serializable
from NN_module.observables import phase_stats_vstate
from NN_module.sampler.sampler import SamplerFactory
from NN_module.saveNload import save_results
from NN_module.schedule.schedule import Schedule
from NN_module.schedule.utils import get_schedule_label
from NN_module.sim_utils import measureNdump
from NN_module.ST_utils import print_tree

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    action="append",
    required=False,
    help="Parse configuration files in order. Simulation/CM/NN",
)

args = parser.parse_args()

if args.config is None:
    args.config = [
        "/home/ihuarte/Escritorio/Ivan/NNs/config0.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config1_CM.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config2_Hydra.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config3_Hydra_NN.json",
    ]
configurations = args.config

print("Configurations:")
for c in configurations:
    print(f" - {c}")

# Cargamos configuraciones de archivos json
with open(configurations[0], "r") as f:
    config = json.load(f)
with open(configurations[1], "r") as f:
    config_cm = json.load(f)
with open(configurations[2], "r") as f:
    config_hydra = json.load(f)
with open(configurations[3], "r") as f:
    config_hydra_nn = json.load(f)

config_nn = {**config_hydra, **config_hydra_nn}


cm_model_name = config_cm["CM"]["selection"]
nn_evol_name = config_nn["selection"]

cm_model_setup = config_cm["CM"][cm_model_name]
nn_evol_setup = config_nn[nn_evol_name]

model_label = cm_model_name + "_" + nn_evol_name

sizes = config_cm["sizes"]

# Simulation settings
schedule_setup = config["schedule"]

### MC sampling rules ###
sampler_setup = config["sampler"]
n_ranks = jax.device_count()
n_samples = (
    n_ranks * sampler_setup["n_samples_per_chain"] * sampler_setup["n_chains_per_rank"]
)
write = get_write_folder_from_model({**config, **config_cm, **config_nn})

for i, size in enumerate(sizes):

    N = int(np.prod(size))
    exact_diag = True if N < 21 else False
    if not exact_diag:
        E_ED = None
        x_ED = None

    write_folder_size = write + f"Size_{size[0]}x{size[1]}/"

    training_folder = get_schedule_label(schedule_setup["learning_rate"])

    write_folder_training = write_folder_size + f"{training_folder}/"

    sim_uuid = str(uuid.uuid4())[:8]
    write_folder = write_folder_training + f"UUID_{sim_uuid}/"

    ###  Reseting Hilbert space object and the observables ###
    hi = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=config["sz_total"])
    model_factory = ModelFactory(size, config_cm)

    for params in model_factory.get_params():

        cm_model_setup = model_factory.get_setup()
        cm_model_name = model_factory.name

        ## Update Hamiltonian
        cm_model = model_factory.get_model()
        eng = Runner(cm_model.cm, S_operators=False)
        H = eng.build_hamiltonian(hi)

        if exact_diag:
            print("Running exact diagonalization...")
            E_ED, x_ED = eng.exact_energy_lanczos(hi, eigenstates=True)
            # x_ED = full_basis_state(x_ED, hi) if hi._total_sz is not None else x_ED
            E_ED = float(E_ED.squeeze(-1))
            print(f"Energy ED: {E_ED}")

        ###################################################

        ######## DISPLAY #########
        sim_config = {
            "SIM": config,
            "CM": cm_model_setup,
            "NN": {"name": nn_evol_name, "setup": nn_evol_setup},
        }
        display_simulation_settings(sim_config)

        #### INITIALIZE NETWORK FRAMEWORK ####
        hydra = Hydra(config_nn, **{"lattice_size": size})
        model = hydra.model
        nparams, nbytes = hydra.n_params, hydra.nbytes
        print(f"Total samples: {n_samples}\n")
        print_tree(hydra.setup, values=True)

        #### INITIALIZE SAMPLER ####
        print("Initializing sampler...")
        sampler_factory = SamplerFactory(sampler_setup, cm_model=cm_model)
        sampler = sampler_factory.get_sampler(hi)

        #### INITIALIZE LOGGER ####
        log = nk.logging.RuntimeLog()
        # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.

        #### INITIALIZE VSTATE ####
        print("Initializing Variational State...")
        seed = int(time.time())
        key = jax.random.key(seed)
        # key = jnp.array([0, 1773936479], dtype=jnp.uint32)
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            sampler_seed=key,
            n_samples=n_samples,
            n_discard_per_chain=0,
            chunk_size=sampler_setup["chunk_vstate"],
        )
        print("Variational state initialized.")

        ## Transplant loaded parameters to the current architecture
        if hydra.load_model:
            vstate = hydra.load_vstate(vstate)
        code2path = hydra.get_code2path(vstate.parameters)

        #### INITIALIZE SCHEDULE ####
        schedule = Schedule(schedule_setup, code2path)
        total_periods = schedule.total_periods
        total_epochs = schedule.total_epochs
        schedule_setup["total_epochs"] = total_epochs

        ds_schedule = jnp.linspace(1e-3, 1e-4, total_periods, dtype=jnp.float64)

        #### INITIALIZE CALLBACKS ####
        if config["callback"]["checkpoint"]:
            sim_label, _, _, _ = get_filenames_from_settings(
                cm_model_setup, {"name": nn_evol_name, "setup": {}}, sim_uuid
            )
            sim_label_folder = write_folder + sim_label
            do_each_checkpoint = config["callback"]["checkpoint_setup"]["do_each"]
        else:
            sim_label_folder = ""
            do_each_checkpoint = None

        callback_objects, callback_funcs = Callback(
            config["callback"],
            sim_config=sim_config,
            total_epochs=total_epochs,
            H=H,
            N=N,
            E_ED=E_ED,
            x_ED=x_ED,
            sim_label_folder=sim_label_folder,
        )

        callback_artifacts = {}
        time_in = time.time()

        for i, (lr_period, info, change) in enumerate(schedule.schedule()):

            #### PRINT PERIOD INFO ####
            epochs = info[0][0]
            mode = info[0][1]
            lr_string = ""
            for inf in info:
                lr_string += f"{inf[2]}  "
            rescaled = "" if len(info) < 4 else info[3]
            print(f"\nPeriod {i + 1} / {total_periods}:")
            print(f"Training {mode} for {epochs} epochs")
            print(f"LR: {lr_string}  ({rescaled})")
            print(f"Diagonal shift: {ds_schedule[i]:.4e}\n")
            ###########################

            optimizer = schedule.transform_optimizer(
                vstate.parameters, optax.sgd, info, lr_period
            )
            # optimizer = optax.adam(0.005)

            if config["vmc_sr"]:

                #### INITIALIZING VMC RUN WITH STOCHASTIC RECONFIGURATION ####
                vmc = nk.driver.VMC_SR(
                    hamiltonian=H.to_jax_operator(),
                    optimizer=optimizer,
                    variational_state=vstate,
                    diag_shift=ds_schedule[i],
                    mode="complex",
                    linear_solver=solver.pinv_smooth
                )
            else:
                #### INITIALIZE OLD VMC WITH SEPARATED SR ####
                sr = nk.optimizer.SR(diag_shift=ds_schedule[i])
                vmc = nk.driver.VMC(
                    H.to_jax_operator(),
                    optimizer=optimizer,
                    variational_state=vstate,
                    preconditioner=sr,
                )

            # import sys
            # sys.exit(0)

            print(f"\nTraining {mode} for {epochs} epochs...")
            vmc.run(
                n_iter=epochs,
                out=log,
                callback=callback_funcs,
                show_progress=True,
            )
            # vstate.sampler.reset(vstate.model.apply, vstate.variables["params"])
            mean, std, psi = phase_stats_vstate(vstate)
            print(f"VS phase: {mean} \u00b1 {std}  ({psi})")


            if change:
                print("Changing Architecture")
                schedule.eon += 1
                hydra.arch_evol(vstate.parameters)  # n_stage + 1
                model = hydra.model
                vstate = nk.vqs.MCState(
                    sampler=sampler,  # sampler_factory.get_sampler(hi),
                    model=model,
                    n_samples=n_samples,
                    n_discard_per_chain=0,
                    chunk_size=sampler_setup["chunk_vstate"],
                )
                vstate.parameters, code2path = hydra.weight_transplantation(
                    vstate.parameters
                )
                schedule.update_eon_code2path(code2path)

        time_out = time.time()
        time_exe = time_out - time_in

        keeper = callback_objects[0]
        best_step = keeper.best_step
        vstate = keeper.best_state

        # Reconstruct NN setup and sim_config in the best state
        eon, _, _ = schedule.locate_period(best_step)
        NN_setup = hydra.storage[f"stage{eon}"]
        sim_config = {
            "SIM": config,
            "CM": cm_model_setup,
            "NN": {"name": nn_evol_name + f"_stage{eon}", "setup": NN_setup},
        }

        if exact_diag:
            keeper.E_ED = E_ED
            keeper.x_ED = x_ED
            log.E_ED = E_ED

        ## Save results

        sim_label, ED_label, json_label, title_label_callback = (
            get_filenames_from_settings(
                cm_model_setup, {"name": nn_evol_name, "setup": {}}, sim_uuid
            )
        )

        # Plot Callback
        callback_args = {
            "size": size,
            "write_folder": write_folder,
            "time_exe": time_exe,
            "sim_label": sim_label,
            "title_label_callback": title_label_callback,
            "best_step": best_step,
            "schedule_setup": schedule.flat_setup(),
            "do_each_checkpoint": do_each_checkpoint,
        }

        callback_artifacts = dump_callback(log, callback_args)

        ## Calculate some observables
        results, mp_array_vs, mp_array_ED = measureNdump(
            keeper,
            time_exe,
            exact_diag=exact_diag,
            S_operators=config_cm["S_operators"],
        )
        results["key"] = jax.random.key_data(key).tolist()
        results["seed"] = seed

        sim_config["SIM"]["sampler"]["nsamples"] = n_samples
        sim_config["SIM"]["sampler"]["final_rng"] = jax.random.key_data(
            vstate.sampler_state.rng
        ).tolist()

        ## Save the results
        dump_setup = {
            **sim_config,
            "results": results,
            "optimizer": "sgd_optax",
            "_artifacts": {
                "callback": callback_artifacts,
            },
        }
        if config["callback"]["checkpoint"]:
            dump_setup["_artifacts"]["checkpoint"] = callback_objects[
                -1
            ].checkpoint_path

        dump_setup["labels"] = {
            "sim_label": sim_label,
            "ED_label": ED_label,
            "json_label": json_label,
            "title_label_callback": title_label_callback,
        }

        dump_setup = make_setup_serializable(dump_setup)

        save_results(
            vstate,
            dump_setup,
            x_ED=x_ED,
            modphase=mp_array_vs,
            modphase_ED=mp_array_ED,
            write_folder=write_folder,
            sim_label=sim_label,
            ED_label=ED_label,
            json_label=json_label,
            sim_uuid=sim_uuid,
        )
