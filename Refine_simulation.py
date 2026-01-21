#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmaz
import optax
import json
import time
import argparse
import uuid
import os

from NN_module.saveNload import load_vstate

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
print(jax.devices())

# Añadir los directorios necesarios
import sys
from pathlib import Path


# Importar módulos necesarios
from NN_module.NN.NN import FactoryBuilder
from VA_project.initialize_model import ModelFactory
from VA_project.engine.runners import Runner

from NN_module.callback.BestIterKeeper import BestIterKeeper
from NN_module.callback.EnergyPlotter import EnergyPlotter
from NN_module.callback.ModPhasePlotter import ModPhasePlotter
from NN_module.callback.SanityMonitor import SanityMonitor

from NN_module.callback.utils import dump_callback

from NN_module.saveNload import save_results
from NN_module.schedule.schedule import Schedule
from NN_module.label_utils import get_filenames_from_settings, get_sim_config

from NN_module.sim_utils import measureNdump
from NN_module.NN_utils import scheduler_initializer
from NN_module.observables import phase_stats_vstate

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--artifact_path",
    type=str,
    required=True,
    help="Path al artefacto principal que recoge los resultados de la simulacion",
)
args = parser.parse_args()
path_artifact = args.artifact_path

with open(path_artifact, "r") as f:
    artifact = json.load(f)

cm_model_name = artifact["CM"]["name"]
nn_model_name = artifact["NN"]["name"]
cm_model_setup = artifact["CM"]
nn_model_setup = artifact["NN"]

##### FILENAMES STUFF ######
# Get filenames in base

sim_label, _, json_label, title_label_callback = get_filenames_from_settings(
    cm_model_setup, nn_model_setup
)

# Refinement filenames
write_folder = os.path.dirname(path_artifact) + "/Refinements/"
if not os.path.exists(write_folder):
    os.makedirs(write_folder)

if os.path.isfile(write_folder + json_label + ".json"):
    i = 1
    file_temp = json_label + f"_{i}.json"
    while os.path.isfile(write_folder + file_temp):
        i += 1
        file_temp = json_label + f"_{i}.json"

    sim_label += f"_{i}"
    json_label += f"_{i}"

############################
# Load refinement configuration

with open("refinement.json", "r") as f:
    config = json.load(f)


# Simulation settings
split_training = config["split_training"]
training_name = "ST_schedule" if split_training else "normal_schedule"
schedule_setup = config[training_name]

### Callbacks
enable_keeper = config["callback"]["keeper"]
enable_inline = config["callback"]["inline"]
enable_modphase = config["callback"]["modphase"]
enable_sanity = config["callback"]["sanity"]
callbacks = []


# Rebuild hamiltonian
size = artifact["CM"]["size"]
N = int(np.prod(size))

####### HAMILTONIAN ########
cm_model = ModelFactory.init(cm_model_setup).get_model()
eng = Runner(cm_model.cm, S_operators=cm_model_setup["S_operators"])
H = eng.build_hamiltonian()


##### VARIATIONAL STATE ######
## Modify whatever in setup

# artifact["NN"]["setup"]["setup"]["phase_setup"]["setup"].pop("Trans_1")

print(f"Loading vstate")
vstate = load_vstate(artifact, only_parameters=True)
# sys.exit(0)
nn_model = vstate.model

print(vstate.model)


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


# Get exact diag energy, in the case.
if artifact["results"]["E_ED"] is not None:
    E_ED = artifact["results"]["E_ED"]
    x_ED = np.loadtxt(artifact["_artifacts"]["x_ED"], dtype=complex)
    exact_diag = True

else:
    E_ED = None
    x_ED = None
    exact_diag = False

# Build a simulation_config
factory = FactoryBuilder(nn_model_setup, **{"lattice_size": size})
model = factory.get_model()
nparams, nbytes = factory.get_params_info(nn_model, N, show_info=False)
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

callback_artifacts = {}
time_in = time.time()

log = (
    nk.logging.RuntimeLog()
)  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.

if split_training:  # Alternated training between modulus and phase

    # SplitTraining Schedule
    schedule = Schedule(schedule_setup, nn_model)
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
        gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)

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
else:

    # Training modulus and phase at the same time
    total_epochs = schedule_setup["total_epochs"]
    # Callbacks
    if enable_keeper:
        keeper = BestIterKeeper(total_epochs, H, N, baseline=1e-8, mode="best_energy")
        callbacks.append(keeper.update)
    # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

    if enable_inline:
        inline_plot = EnergyPlotter(H, N, E_ED=E_ED)
        callbacks.append(inline_plot)
    schedule_setup["total_epochs"] = total_epochs

    ds_schedule = optax.linear_schedule(1e-2, 1e-4, total_epochs)
    SR = nk.optimizer.SR(diag_shift=ds_schedule)
    lr_schedule = scheduler_initializer("warmup_exponential_decay", schedule_setup)
    optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=SR).run(
        n_iter=total_epochs,
        out=log,
        callback=callbacks,
        show_progress=True,
    )

time_out = time.time()
time_exe = time_out - time_in

vstate = keeper.best_state

resp = input("Do you want to save simulation? [y/N]: ").strip().lower()
if resp not in ("y", "s", "si", "yes"):
    print("Execution stopped")
    sys.exit(0)
print("Saving....")


if exact_diag:
    keeper.E_ED = E_ED
    keeper.x_ED = x_ED
    log.E_ED = E_ED

# Plot Callback
dump_setup = {
    "size": size,
    "write_folder": write_folder,
    "time_exe": time_exe,
    "sim_label": sim_label,
    "title_label_callback": title_label_callback,
    "best_step": keeper.best_step,
    "schedule_setup": {
        "lr_name": lr_name,
        **schedule_setup["setup"],
        **schedule_setup["lr_schedules"][lr_name],
    },
}
callback_artifacts = dump_callback(log, dump_setup)

## Calculate some observables
results, mp_array_vs, mp_array_ED = measureNdump(keeper, time_exe, exact_diag)

# Save the results
artifact["SIM"]["sampler"]["rng"] = vstate.sampler_state.rng.tolist()
artifact["SIM"]["schedule"]["lr_schedule"]["name"] = lr_name
artifact["SIM"]["schedule"]["lr_schedule"]["setup"] = schedule_setup["lr_schedules"][
    lr_name
]
artifact["SIM"]["schedule"]["setup"] = schedule_setup

artifact["results"] = results

artifact["_artifacts"]["callback"] = callback_artifacts

save_results(
    vstate,
    artifact,
    x_ED=None,
    modphase=mp_array_vs,
    modphase_ED=None,
    write_folder=write_folder,
    sim_label=sim_label,
    ED_label=None,
    json_label=json_label,
)
