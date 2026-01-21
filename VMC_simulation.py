#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import optax
import json
import time
import argparse
import uuid
import sys

import matplotlib

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
print(jax.devices())

# Añadir los directorios necesarios
import sys
from pathlib import Path

# Importar módulos necesarios
from VA_project.initialize_model import ModelFactory
from VA_project.engine.runners import Runner
from NN_module.callback.BestIterKeeper import BestIterKeeper
from NN_module.callback.EnergyPlotter import EnergyPlotter
from NN_module.callback.ModPhasePlotter import ModPhasePlotter
from NN_module.callback.SanityMonitor import SanityMonitor
from NN_module.callback.utils import dump_callback

from NN_module.saveNload import save_results
from NN_module.NN.NN import FactoryBuilder, print_tree
from NN_module.sampler.sampler import SamplerFactory
from NN_module.schedule.schedule import Schedule
from NN_module.label_utils import (
    get_filenames_from_settings,
    get_write_folder_from_model,
    display_simulation_settings,
    get_sim_config,
)
from NN_module.schedule.utils import get_schedule_label
from NN_module.sim_utils import measureNdump
from NN_module.observables import full_basis_state, phase_stats_vstate
from NN_module.ST_utils import compare_params

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
        "/home/ihuarte/Escritorio/Ivan/NNs/config.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config_CM.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config_NN.json",
    ]
configurations = args.config

print(f"Configurations:")
for c in configurations:
    print(f" - {c}")

# Cargamos configuraciones de archivos json
with open(configurations[0], "r") as f:
    config = json.load(f)
with open(configurations[1], "r") as f:
    config_cm = json.load(f)
with open(configurations[2], "r") as f:
    config_nn = json.load(f)

cm_model_name = config_cm["CM"]["selection"]
nn_model_name = config_nn["NN"]["selection"]

cm_model_setup = config_cm["CM"][cm_model_name]
nn_model_setup = config_nn["NN"][nn_model_name]

model_label = cm_model_name + "_" + nn_model_name

sizes = config_cm["sizes"]

# Simulation settings
schedule_setup = config["schedule"]["learning_rate"]

### MC sampling rules ###
sampler_setup = config["sampler"]
n_samples = (
    sampler_setup["n_samples_per_chain"]
    * sampler_setup["n_chains_per_rank"]
    * sampler_setup["n_ranks"]
)

### Callbacks
enable_keeper = config["callback"]["keeper"]
enable_inline = config["callback"]["inline"]
enable_modphase = config["callback"]["modphase"]
enable_sanity = config["callback"]["sanity"]
callbacks = []

write = get_write_folder_from_model({**config, **config_cm, **config_nn})

for i, size in enumerate(sizes):

    N = int(np.prod(size))
    exact_diag = True if N < 21 else False
    if not exact_diag:
        E_ED = None
        x_ED = None

    write_folder_size = write + f"Size_{size[0]}x{size[1]}/"

    training_folder = get_schedule_label(schedule_setup)

    write_folder_training = write_folder_size + f"{training_folder}/"

    sim_uuid = str(uuid.uuid4())[:8]
    write_folder = write_folder_training + f"UUID_{sim_uuid}/"

    ###  Reseting Hilbert space object and the observables ###
    hi = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=None)
    model_factory = ModelFactory(size, config_cm)

    for params in model_factory.get_params():

        cm_model_setup = model_factory.get_setup()
        cm_model_name = model_factory.name

        ## Update Hamiltonian
        cm_model = model_factory.get_model()
        eng = Runner(cm_model.cm, S_operators=model_factory.S_operators)
        H = eng.build_hamiltonian(hi)

        if exact_diag:
            print("Running exact diagonalization...")
            E_ED, x_ED = eng.exact_energy_lanczos(hi, eigenstates=True)
            # x_ED = full_basis_state(x_ED, hi) if hi._total_sz is not None else x_ED
            E_ED = float(E_ED.squeeze(-1))
            print(f"Energy ED: {E_ED}")

        ###################################################

        factory = FactoryBuilder(nn_model_setup, **{"lattice_size": size})
        model = factory.get_model()
        nparams, nbytes = factory.get_params_info(model, N, show_info=False)

        sim_config = get_sim_config(
            {
                "SIM": config,
                "CM": cm_model_setup,
                "NN": {
                    "name": nn_model_name,
                    "n_params": nparams,
                    "nbytes": nbytes,
                    "setup": nn_model_setup,
                },
            }
        )
        # print_tree(sim_config, values=True)

        ## Reset sampler
        print("Initializing sampler...")
        sampler = SamplerFactory(sampler_setup, cm_model=cm_model).get_sampler(hi)

        display_simulation_settings({**sim_config})
        print(f"\nNN stats: {nparams} parameters ({nbytes/(1024**2)} MB)")
        print(f"Total samples: {n_samples}\n")

        callback_artifacts = {}
        time_in = time.time()

        log = (
            nk.logging.RuntimeLog()
        )  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.
        print("Initializing VMC state...")
        vstate = nk.vqs.MCState(
            sampler,
            model=model,
            n_samples=n_samples,
            n_discard_per_chain=0,
            chunk_size=sampler_setup["chunk_vstate"],
        )
        sys.exit(0)

        # SplitTraining Schedule
        schedule = Schedule(schedule_setup, model)
        total_periods = schedule.total_periods
        total_epochs = schedule.total_epochs
        schedule_setup["total_epochs"] = total_epochs

        ds_schedule = jnp.linspace(1e-2, 1e-4, total_periods, dtype=jnp.float64)

        # Callbacks
        if enable_keeper:
            keeper = BestIterKeeper(total_epochs, H, N, baseline=1e-8, mode="always")
            callbacks.append(keeper.update)
        if enable_inline:
            inline_energy = EnergyPlotter(H, N, E_ED=E_ED)
            callbacks.append(inline_energy)
        if enable_modphase:
            print(f"Adding Modphase")
            inline_modphase = ModPhasePlotter(sim_config, x_ED)
            callbacks.append(inline_modphase)
        if enable_sanity:
            sanity_monitor = SanityMonitor(config["callback"]["sanity_setup"])
            callbacks.append(sanity_monitor)

        for i, (lr_period, info) in enumerate(schedule.schedule()):
            print(f"info: {info}")

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

            variables = vstate.variables
            sampler = vstate.sampler
            optimizer = schedule.transform_optimizer(
                vstate.parameters, optax.sgd, info, lr_period
            )
            sys.exit(0)

            # vstate = nk.vqs.MCState(
            #     sampler,
            #     sampler_seed=vstate.sampler_state.rng,
            #     model=model,
            #     n_samples=n_samples,
            #     n_discard_per_chain=0,
            #     chunk_size=sampler_setup["chunk_vstate"],
            #     variables=variables,
            # )
            if i != 0:
                vstate = nk.vqs.MCState(
                    sampler,
                    model=model,
                    n_samples=n_samples,
                    n_discard_per_chain=500,
                    chunk_size=sampler_setup["chunk_vstate"],
                    variables=variables,
                )
            holo = nk.utils.is_probably_holomorphic(
                vstate._apply_fun,
                vstate.parameters,
                vstate.samples,
                model_state=vstate.model_state,
            )
            sr = nk.optimizer.SR(diag_shift=ds_schedule[i], holomorphic=False)
            gs = nk.driver.VMC(
                H, optimizer, variational_state=vstate, preconditioner=sr
            )

            print(f"\nTraining {mode} for {epochs} epochs...")
            gs.run(
                n_iter=epochs,
                out=log,
                callback=callbacks,
                show_progress=True,
            )
            # vstate.sampler.reset(vstate.model.apply, vstate.variables["params"])
            mean, std, psi = phase_stats_vstate(vstate)
            print(f"VS phase: {mean} \u00b1 {std}  ({psi})")

            # P1 = vstate.parameters
            # compare_params(P0, P1)
            # check_zero_grads(vstate, mask)

        time_out = time.time()
        time_exe = time_out - time_in

        vstate = keeper.best_state

        if exact_diag:
            keeper.E_ED = E_ED
            keeper.x_ED = x_ED
            log.E_ED = E_ED

        ## Save results

        sim_label, ED_label, json_label, title_label_callback = (
            get_filenames_from_settings(
                cm_model_setup, {"name": nn_model_name, **nn_model_setup}, sim_uuid
            )
        )

        # Plot Callback
        callback_args = {
            "size": size,
            "write_folder": write_folder,
            "time_exe": time_exe,
            "sim_label": sim_label,
            "title_label_callback": title_label_callback,
            "best_step": keeper.best_step,
            "schedule_setup": schedule.flat_setup(),
        }

        callback_artifacts = dump_callback(log, callback_args)

        ## Calculate some observables
        results, mp_array_vs, mp_array_ED = measureNdump(keeper, time_exe, exact_diag)

        sim_config["SIM"]["sampler"]["nsamples"] = n_samples
        sim_config["SIM"]["sampler"]["rng"] = jax.random.key_data(
            vstate.sampler_state.rng
        ).tolist()

        ## Save the results
        dump_setup = {
            **sim_config,
            "results": results,
            "optimizer": "Sgd",
            "_artifacts": {"callback": callback_artifacts},
        }

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
