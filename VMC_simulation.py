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
from NN_module.callbacks import (
    BestIterKeeper,
    EnergyPlotter,
    ModPhasePlotter,
    dump_callback,
)
from NN_module.saveNload import save_results
from NN_module.initialize_NN import FactoryBuilder, print_tree
from NN_module.initialize_sampler import SamplerFactory
from NN_module.schedules import generate_training
from NN_module.label_utils import (
    get_filenames_from_settings,
    get_write_folder_from_model,
    display_simulation_settings,
    get_ST_folder,
    get_sim_config,
)
from NN_module.sim_utils import measureNdump
from NN_module.NN_utils import scheduler_initializer, phase_stats_vstate
from NN_module.ST_utils import masked_optimizer

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
split_training = config["split_training"]
training_name = "ST_schedule" if split_training else "normal_schedule"
lr_name = config[training_name]["lr_name"]
training_setup = config[training_name]

sampler_setup = config["sampler"]
n_samples = (
    sampler_setup["n_samples_per_chain"]
    * sampler_setup["n_chains_per_rank"]
    * sampler_setup["n_ranks"]
)
print(f"Total samples: {n_samples}")

### MC sampling rules ###
sampler_setup = config["sampler"]

### Callbacks
enable_keeper = config["callback"]["keeper"]
enable_inline = config["callback"]["inline"]
enable_modphase = config["callback"]["modphase"]
callbacks = []


write = get_write_folder_from_model({**config, **config_cm, **config_nn})

for i, size in enumerate(sizes):

    N = int(np.prod(size))
    exact_diag = True if N < 21 else False
    if not exact_diag:
        E_ED = None
        x_ED = None

    write_folder_size = write + f"Size_{size[0]}x{size[1]}/"

    training_folder = get_ST_folder(split_training, training_setup)

    write_folder_training = write_folder_size + f"{training_folder}/"

    sim_uuid = str(uuid.uuid4())[:8]
    write_folder = write_folder_training + f"UUID_{sim_uuid}/"

    ###  Reseting Hilbert space object and the observables ###
    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    ## Reset sampler
    sampler = SamplerFactory(sampler_setup).get_sampler(hi)
    model_factory = ModelFactory(size, config_cm)

    for params in model_factory.get_params():

        cm_model_setup = model_factory.get_setup()
        cm_model_name = model_factory.name

        sim_config = get_sim_config(
            {
                "SIM": config,
                "CM": {"name": cm_model_name, "setup": cm_model_setup},
                "NN": {"name": nn_model_name, "setup": nn_model_setup},
            }
        )
        # print_tree(sim_config, values=True)

        display_simulation_settings({**sim_config})

        ## Update Hamiltonian
        cm_model = model_factory.get_model()
        eng = Runner(cm_model.cm, S_operators=model_factory.S_operators)
        H = eng.build_hamiltonian()

        if exact_diag:
            print("Running exact diagonalization...")
            E_ED, x_ED = eng.exact_energy_lanczos(eigenstates=True)
            E_ED = float(E_ED.squeeze(-1))
            print(f"Energy ED: {E_ED}")

        ###################################################

        callback_artifacts = {}
        time_in = time.time()

        factory = FactoryBuilder(nn_model_setup, **{"lattice_size": size})
        model = factory.get_model()

        log = (
            nk.logging.RuntimeLog()
        )  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.

        # Initialize vstate with parameters
        vstate = nk.vqs.MCState(
            sampler,
            model=model,
            n_samples=n_samples,
            n_discard_per_chain=0,
            chunk_size=sampler_setup["chunk_vstate"],
        )

        if split_training:  # Alternated training between modulus and phase

            segments, modes, lr_segments = generate_training(training_setup)

            total_segments = len(segments)
            total_epochs = int(np.array([s for seg in segments for s in seg]).sum())
            training_setup["total_epochs"] = total_epochs

            ds_schedule = jnp.linspace(1e-2, 1e-4, total_segments)

            transformations = {
                "train": optax.sgd(0.1),
                "freeze": optax.set_to_zero(),
            }

            # Callbacks
            if enable_keeper:
                keeper = BestIterKeeper(
                    total_epochs, H, N, baseline=1e-8, mode="best_energy"
                )
                callbacks.append(keeper.update)
            # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.
            if enable_inline:
                inline_energy = EnergyPlotter(H, N, E_ED=E_ED)
                callbacks.append(inline_energy)
            if enable_modphase:
                inline_modphase = ModPhasePlotter(sim_config, x_ED)
                callbacks.append(inline_modphase)

            print(f"\nEpochs:      {total_epochs}")
            print(f"Segments:      {segments}")
            print(f"Modes:         {modes}")

            for i, (segment, seg_modes, lr_segment) in enumerate(
                zip(segments, modes, lr_segments)
            ):

                SR = nk.optimizer.SR(diag_shift=ds_schedule[i])
                print(f"\nSegment:")
                print(f"  Epochs: {segment}")
                print(f"  Modes:  {seg_modes}")
                print(f"  LRs:    {lr_segment}")

                for epochs, mode, lr in zip(segment, seg_modes, lr_segment):

                    print(
                        f"\nSegment {i+1} of {total_segments}......   lr: {lr:.4e}  ds: {ds_schedule[i]:.4e}\n"
                    )
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
                            f"'M' for modulus, 'P' for phase and 'B' for both"
                        )

                    variables = vstate.variables
                    sampler = vstate.sampler
                    optimizer = masked_optimizer(
                        vstate.parameters, transformations, mode=mask
                    )

                    vstate = nk.vqs.MCState(
                        sampler,
                        sampler_seed=vstate.sampler_state.rng,
                        model=model,
                        n_samples=n_samples,
                        n_discard_per_chain=0,
                        chunk_size=sampler_setup["chunk_vstate"],
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
                keeper = BestIterKeeper(
                    total_epochs, H, N, baseline=1e-8, mode="best_energy"
                )
                callbacks.append(keeper.update)
            # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

            if enable_inline:
                inline_plot = EnergyPlotter(H, N, E_ED=E_ED)
                callbacks.append(inline_plot)
            training_setup["total_epochs"] = total_epochs

            ds_schedule = optax.linear_schedule(1e-2, 1e-4, total_epochs)
            SR = nk.optimizer.SR(diag_shift=ds_schedule)
            lr_schedule = scheduler_initializer(
                "warmup_exponential_decay", training_setup
            )
            optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
            gs = nk.driver.VMC(
                H, optimizer, variational_state=vstate, preconditioner=SR
            ).run(
                n_iter=total_epochs,
                out=log,
                callback=callbacks,
                show_progress=True,
            )

        time_out = time.time()
        time_exe = time_out - time_in

        vstate = keeper.best_state

        if exact_diag:
            keeper.E_ED = E_ED
            keeper.x_ED = x_ED
            log.E_ED = E_ED

        ## Save results

        sim_label, ED_label, json_label, title_label_callback = (
            get_filenames_from_settings(nn_model_setup, cm_model_setup, sim_uuid)
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
                **training_setup["setup"],
                **training_setup["lr_schedules"][lr_name],
            },
        }

        callback_artifacts = dump_callback(log, dump_setup)

        ## Calculate some observables
        results, mp_array_vs, mp_array_ED = measureNdump(keeper, time_exe, exact_diag)

        ## Save the results
        dump_setup = {
            **sim_config,
            "sampler": {
                "name": "MetropolisSampler",
                "n_samples": n_samples,
                "rng": vstate.sampler_state.rng.tolist(),
                "rules": "LocalRule/InvertMagnetization",
                "setup": sampler_setup,
            },
            "optimizer": "Sgd",
            "results": {**results},
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
