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
from VA_project.model.model import J1J2Square
from VA_project.engine.runners import Runner
from NN_module.callbacks import (
    BestIterKeeper,
    EnergyPlotter,
    ModPhasePlotter,
    dump_callback,
)
from NN_module.saveNload import save_results
from NN_module.initialize_NN import FactoryBuilder
from NN_module.schedules import get_ST_schedule
from NN_module.label_utils import get_filenames_from_settings

from NN_module.sim_utils import measureNdump
from NN_module.NN_utils import scheduler_initializer, phase_stats_vstate, modphase
from NN_module.ST_utils import check_zero_grads, compare_params, masked_optimizer
from NN_module.observables import calc_all_observables_vs, calc_all_observables_ED
from NN_module.initialize_sampler import InvertMagnetization, LocalRule_Z2

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

##### FILENAMES STUFF ######
# Get filenames in base
kwargs = {
    "size": artifact["lattice"]["size"],
    **artifact["CM"],
    **artifact["NN"]["setup"],
}
sim_label, _, json_label, title_label_callback = get_filenames_from_settings(
    "Oxalate", artifact["NN"]["name"], **kwargs
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
lr_name = config[training_name]["lr_name"]
training_setup = config[training_name]["setup"]
lr_schedule_setup = config[training_name]["lr_schedules"][lr_name]

### Callbacks
enable_keeper = config["callback"]["keeper"]
enable_inline = config["callback"]["inline"]
enable_modphase = config["callback"]["modphase"]
callbacks = []

# Load vstate....
print(f"Loading vstate and burning 1000 samples")
vstate = load_vstate(artifact)
# for i in range(1000):
#     print(f"Recalentando samples: {i}")
#     vstate.sample()

# Rebuild hamiltonian
size = artifact["lattice"]["size"]
N = int(np.prod(size))
J1 = artifact["CM"]["J1"]
J2 = artifact["CM"]["J2"]
fields = artifact["CM"]["fields"]


j1j2 = J1J2Square(
    size,
    J1,
    J2,
    fields,
    **{"bc": artifact["lattice"]["bc"], "order": artifact["lattice"]["order"]},
)
eng = Runner(j1j2.cm)
H = eng.build_hamiltonian()

# Get exact diag energy, in the case.

if artifact["results"]["E_ED"] is not None:
    E_ED = artifact["results"]["E_ED"]
    x_ED = np.loadtxt(artifact["results"]["xED"], dtype=complex)
    exact_diag = True

else:
    E_ED = None
    x_ED = None
    exact_diag = False


callback_artifacts = {}
time_in = time.time()

factory = FactoryBuilder(artifact["NN"]["setup"], **{"lattice_size": size})
model = factory.get_model()

log = (
    nk.logging.RuntimeLog()
)  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.

if split_training:  # Alternated training between modulus and phase

    segments = []
    modes = []
    s, m, r = (
        training_setup["segments"],
        training_setup["mode"],
        training_setup["repeat_segment"],
    )

    for seg, mode, repeats in zip(s, m, r):
        segments += [seg] * repeats
        modes += [mode] * repeats

    total_segments = len(segments)
    total_epochs = int(np.array([s for seg in segments for s in seg]).sum())
    training_setup["total_epochs"] = total_epochs

    lr_schedule = get_ST_schedule(
        lr_name,
        {
            **lr_schedule_setup,
            "total_epochs": total_epochs,
            "total_segments": total_segments,
        },
    )
    ds_schedule = jnp.linspace(1e-2, 1e-4, total_segments)

    transformations = {
        "train": optax.sgd(0.1),
        "freeze": optax.set_to_zero(),
    }

    # Callbacks
    if enable_keeper:
        keeper = BestIterKeeper(total_epochs, H, N, baseline=1e-8, mode="best_energy")
        callbacks.append(keeper.update)
    # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.
    if enable_inline:
        inline_energy = EnergyPlotter(
            H,
            N,
            artifact["results"]["E_best"],
            artifact["results"]["E_ED"],
            artifact["results"]["vscore"],
            artifact["results"]["error"],
        )
        callbacks.append(inline_energy)
    if enable_modphase:
        inline_modphase = ModPhasePlotter(artifact, x_ED)
        callbacks.append(inline_modphase)

    print(f"\nEpochs:      {total_epochs}")
    print(f"Segments:      {segments}")
    print(f"Modes:         {modes}")

    for i, (segment, seg_modes, lr) in enumerate(zip(segments, modes, lr_schedule)):

        print(
            f"\nSegment {i+1} of {total_segments}......   lr: {lr:.4f}  ds: {ds_schedule[i]:.4f}\n"
        )
        SR = nk.optimizer.SR(diag_shift=ds_schedule[i])

        for epochs, mode in zip(segment, seg_modes):

            # P0 = vstate.parameters

            if mode == "M":
                mode = "modulus"
                mask = "phase"
                transformations["train"] = optax.sgd(learning_rate=lr)

            elif mode == "P":
                mode = "phase"
                mask = "modulus"
                transformations["train"] = optax.sgd(learning_rate=lr)

            elif mode == "B":
                mode = "both"
                mask = None
                transformations["train"] = optax.sgd(learning_rate=lr)

            else:
                raise ValueError(
                    f"Invalid training mode: {mode}."
                    f"'M' for modulus and 'P' for phase"
                )

            variables = vstate.variables
            sampler = vstate.sampler
            optimizer = masked_optimizer(vstate.parameters, transformations, mode=mask)

            vstate = nk.vqs.MCState(
                sampler,
                sampler_seed=vstate.sampler_state.rng,
                model=model,
                n_samples=artifact["sampler"]["n_samples"],
                n_discard_per_chain=0,
                chunk_size=artifact["sampler"]["chunk_vstate"],
                variables=variables,
            )

            gs = nk.driver.VMC(
                H,
                optimizer,
                variational_state=vstate,
                preconditioner=SR,
            )

            print(f"\nTraining {mode} for {epochs} epochs...")
            gs.run(
                n_iter=epochs,
                out=log,
                callback=callbacks,
                show_progress=True,
            )
            mean, std, psi = phase_stats_vstate(vstate)
            print(f"VS phase: {mean} \u00b1 {std}  ({psi})")
            # P1 = vstate.parameters
            # print(compare_params(P0, P1))
            # check_zero_grads(vstate, mask)

else:  # Training modulus and phase at the same time

    total_epochs = training_setup["total_epochs"]
    # Callbacks
    if enable_keeper:
        keeper = BestIterKeeper(total_epochs, H, N, baseline=1e-8, mode="best_energy")
        callbacks.append(keeper.update)
    # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

    if enable_inline:
        inline_plot = EnergyPlotter(H, N, E_ED=E_ED)
        callbacks.append(inline_plot)
    lr_schedule_setup["total_epochs"] = total_epochs

    ds_schedule = optax.linear_schedule(1e-2, 1e-4, total_epochs)
    SR = nk.optimizer.SR(diag_shift=ds_schedule)
    lr_schedule = scheduler_initializer("warmup_exponential_decay", lr_schedule_setup)
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

## Save results
_kwargs = artifact["NN"][artifact["NN"]["name"]].copy()
_kwargs["size"] = size
_kwargs["J1"] = J1
_kwargs["J2"] = J2
_kwargs["fields"] = fields

sim_label, ED_label, json_label, title_label_callback = get_filenames_from_settings(
    config["CM"]["selection"], config["NN"]["selection"], **_kwargs
)

# Plot Callback
dump_setup = {
    "size": size,
    "write_folder": write_folder,
    "time_exe": time_exe,
    "sim_label": sim_label,
    "title_label_callback": title_label_callback,
    "best_step": keeper.best_step,
    "training_setup": {
        "lr_name": lr_name,
        **training_setup,
        **lr_schedule_setup,
    },
}
callback_artifacts = dump_callback(log, dump_setup)

## Calculate some observables
results, mp_array_vs, mp_array_ED = measureNdump(keeper, time_exe, exact_diag)

# Save the results
artifact["SIM"]["sampler"]["rng"] = vstate.sampler_state.rng.tolist()
artifact["SIM"]["lr_schedule"]["name"] = lr_name
artifact["SIM"]["lr_schedule"]["setup"] = lr_schedule_setup
artifact["SIM"]["setup"] = training_setup

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
