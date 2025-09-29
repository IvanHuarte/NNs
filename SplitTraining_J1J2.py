#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from netket.operator.spin import sigmaz
import optax
import json
import time
import ast
import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()

# Añadir los directorios necesarios
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA/VA_project/src"))
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent / "Transformers/transformer_LR_WF_public"
    )
)

# Importar módulos necesarios
from VA_project.model.model import J1J2Square
from VA_project.engine.runners import Runner
from NN_module.sim_utils import save_results, dump_callback, init_model, BestIterKeeper
from NN_module.label_utils import (
    get_filenames_from_settings,
    architecture_label,
    get_write_folder_from_model,
    display_simulation_settings,
    get_ST_folder,
)
from NN_module.schedules import get_ST_schedule
from NN_module.NN_utils import scheduler_initializer, phase_stats_vstate, modphase
from NN_module.ST_utils import check_zero_grads, compare_params, masked_optimizer
from NN_module.observables import calc_all_observables_vs, calc_all_observables_ED
from transformer_LR_WF.utils import InvertMagnetization

# Cargamos configuraciones de archivos json
with open("/home/ihuarte/Escritorio/Ivan/NNs/config.json", "r") as f:
    config = json.load(f)
with open("/home/ihuarte/Escritorio/Ivan/NNs/config_CM.json", "r") as f:
    config_cm = json.load(f)
with open("/home/ihuarte/Escritorio/Ivan/NNs/config_NN.json", "r") as f:
    config_nn = json.load(f)


cm_model_name = config_cm["CM"]["selection"]
nn_model_name = config_nn["model_NN"]["selection"]

nn_model_setup = config_nn["model_NN"][nn_model_name]
model_label = cm_model_name + "_" + nn_model_name

sizes = config_cm["sizes"]
J1_list = config_cm["CM"][cm_model_name]["J1_list"]  # Lattice and coupling model
J2_list = config_cm["CM"][cm_model_name]["J2_list"]  # Lattice and coupling model
fields_list = config_cm["CM"][cm_model_name]["fields_list"]
kwargs_lattice = config_cm["kwargs_lattice"]

# Simulation settings
split_training = config["split_training"]
training_name = "ST_schedule" if split_training else "normal_schedule"
lr_name = config[training_name]["lr_name"]
training_setup = config[training_name]["setup"]
lr_schedule_setup = config[training_name]["lr_schedules"][lr_name]

exact_diag = config["exact_diagonalization"]
dump_simulation = config["dump_sim_callback"]

sampler_setup = config["sampler"]
n_samples = (
    sampler_setup["n_samples_per_chain"]
    * sampler_setup["n_chains_per_rank"]
    * sampler_setup["n_ranks"]
)
print(f"Total samples: {n_samples}")

write = get_write_folder_from_model({**config, **config_cm, **config_nn})

### MC sampling rules ###
rule1 = nk.sampler.rules.LocalRule()
rule2 = InvertMagnetization()
pinvert = 0.25
pflip = 1 - pinvert

E_ED = None
x_ED = None

for i, size in enumerate(sizes):
    nn_model_setup["lattice_size"] = size

    N = int(np.prod(size))
    if N > 20:
        exact_diag = False

    write_folder_size = write + f"Size_{size[0]}x{size[1]}/"

    training_folder = get_ST_folder(split_training, training_setup)

    write_folder = write_folder_size + f"{training_folder}/"

    ###  Reseting Hilbert space object and the observables ###
    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    ## Reset sampler
    sampler = nk.sampler.MetropolisSampler(
        hi,
        nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert]),
        n_chains_per_rank=sampler_setup["n_chains_per_rank"],
        chunk_size=sampler_setup["chunk_sampler"],
    )

    for fields in fields_list:

        for J1, J2 in zip(J1_list, J2_list):

            config_cm["CM"][cm_model_name]["J1"] = J1
            config_cm["CM"][cm_model_name]["J2"] = J2
            config_cm["CM"][cm_model_name]["fields"] = fields
            config_cm["size"] = size
            display_simulation_settings({**config_cm, **config_nn})

            ## Update Hamiltonian

            j1j2 = J1J2Square(size, J1, J2, fields, **kwargs_lattice)

            eng = Runner(j1j2.cm, S_operators=False)
            H = eng.build_hamiltonian()

            if exact_diag:
                print("Running exact diagonalization...")
                E_ED, x_ED = eng.exact_energy_lanczos(eigenstates=True)
                E_ED = float(E_ED.squeeze(-1))
                print(f"Energy ED: {E_ED}")

            ####################################################

            callback_artifacts = {}
            time_in = time.time()

            model = init_model(nn_model_name, nn_model_setup)

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
                ds_schedule = jnp.linspace(1e-2, 1e-4, total_epochs)

                transformations = {
                    "train": optax.sgd(0.1),
                    "freeze": optax.set_to_zero(),
                }

                keeper = BestIterKeeper(
                    total_epochs, H, N, baseline=1e-8, mode="best_energy"
                )
                # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

                print(f"\nEpochs:      {total_epochs}")
                print(f"Segments:    {segments}")
                print(f"Modes:       {modes}")

                for i, (segment, seg_modes, lr) in enumerate(
                    zip(segments, modes, lr_schedule)
                ):

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
                            transformations["train"] = optax.sgd(learning_rate=10 * lr)

                        else:
                            raise ValueError(
                                f"Invalid training mode: {mode}."
                                f"'M' for modulus and 'P' for phase"
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
                            callback=[keeper.update],
                            show_progress=True,
                        )
                        mean, std, psi = phase_stats_vstate(vstate)
                        print(f"VS phase: {mean} \u00b1 {std}  ({psi})")

                        # P1 = vstate.parameters
                        # print(compare_params(P0,P1))
                        # check_zero_grads(vstate, mask)

            else:  # Training modulus and phase at the same time

                total_epochs = training_setup["total_epochs"]
                keeper = BestIterKeeper(
                    total_epochs, H, N, baseline=1e-8, mode="best_energy"
                )
                # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.
                lr_schedule_setup["total_epochs"] = total_epochs

                ds_schedule = optax.linear_schedule(1e-2, 1e-4, total_epochs)
                SR = nk.optimizer.SR(diag_shift=ds_schedule)
                lr_schedule = scheduler_initializer(
                    "warmup_exponential_decay", lr_schedule_setup
                )
                optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
                gs = nk.driver.VMC(
                    H, optimizer, variational_state=vstate, preconditioner=SR
                ).run(
                    n_iter=total_epochs,
                    out=log,
                    callback=[keeper.update],
                    show_progress=True,
                )

            time_out = time.time()
            time_exe = time_out - time_in

            if exact_diag:
                keeper.E_ED = E_ED
                log.E_ED = E_ED

            ## Save results
            _kwargs = config_nn["model_NN"][nn_model_name]
            _kwargs["size"] = size
            _kwargs["J1"] = J1
            _kwargs["J2"] = J2
            _kwargs["fields"] = fields

            sim_label, ED_label, json_label, title_label_callback = (
                get_filenames_from_settings(
                    config_cm["CM"]["selection"],
                    config_nn["model_NN"]["selection"],
                    **_kwargs,
                )
            )

            if dump_simulation:
                # For plotting architecture
                architecture = architecture_label(nn_model_name, nn_model_setup)
                dump_setup = {
                    "size": size,
                    "opt_name": "Sgd",
                    "learning_rate": "Scheduled",
                    "write_folder": write_folder,
                    "time_exe": time_exe,
                    "architecture": architecture,
                    "sim_label": sim_label,
                    "title_label_callback": title_label_callback,
                    "best_step": keeper.best_step,
                }

                callback_artifacts = dump_callback(log, dump_setup)

            else:
                callback_artifacts = None

            ## Calculate some observables
            # Modulus and phase

            vstate = keeper.best_state
            best_step = keeper.best_step
            E_best = float(keeper.best_state_energy)
            vscore = float(keeper.best_state_vscore)

            modphase_results = {}
            if exact_diag:
                error = float(np.abs(E_best - E_ED) / np.abs(E_ED))
                mp_array_ED, stats_ED = modphase(x_ED)
                modphase_results["xED"] = stats_ED
                print(
                    f"xED phase: {stats_ED['phase']['mean']} \u00b1 {stats_ED['phase']['std']}  ({stats_ED['type']})"
                )

            else:
                E_ED = None
                x_ED = None
                error = None

            mp_array_vs, stats_vs = modphase(vstate)
            modphase_results["vstate"] = stats_vs
            print(
                f"vstate phase: {stats_vs['phase']['mean']} \u00b1 {stats_vs['phase']['std']}  ({stats_vs['type']})"
            )

            # Fidelity
            fidelity = None
            try:
                fidelity = float(jnp.abs(jnp.vdot(vstate.to_array(), x_ED.squeeze())))
                print(f"Fidelity: {fidelity:.3e}")
            except (MemoryError, RuntimeError, ValueError):
                print(f"Failed fidelity calculation due to memory allocation error")

            # Renyi entropy, magnetization and its fluctuation
            S_renyi, m, ms, m2, ms2 = calc_all_observables_vs(vstate)

            print(f"\nRenyi entropy: {S_renyi}")
            print(f"< m >: {m}   < m2 >: {m2}")
            print(f"< ms >: {ms}  < ms2 >: {ms2}")

            if exact_diag:
                m_ED, ms_ED, m2_ED, ms2_ED = calc_all_observables_ED(x_ED.squeeze())
                print(f"ED < m >: {m_ED}   < m2 >: {m2_ED}")
                print(f"ED < ms >: {ms_ED}  < ms2 >: {ms2_ED}\n")

            else:
                m_ED, ms_ED, m2_ED, ms2_ED = None

            ## Save the results
            dump_setup = {
                "model_label": model_label,
                "lattice": {
                    "name": cm_model_name,
                    "size": size,
                    "bc": kwargs_lattice["bc"],
                    "order": kwargs_lattice["order"],
                },
                "coupling_model": {
                    "J1": J1,
                    "J2": J2,
                    "fields": fields,
                },
                "model_NN": {
                    "name": nn_model_name,
                    "setup": nn_model_setup,
                },
                "sampler": {
                    "name": "MetropolisSampler",
                    "n_samples": n_samples,
                    "rng": vstate.sampler_state.rng.tolist(),
                    "rules": "LocalRule/InvertMagnetization",
                    "setup": sampler_setup,
                },
                "optimizer": "Sgd",
                "training": {
                    "name": training_name,
                    "setup": training_setup,
                    "lr_schedule": {"name": lr_name, "setup": lr_schedule_setup},
                },
                "results": {
                    "best_step": best_step,
                    "E_best": E_best,
                    "E_ED": E_ED,
                    "error": error,
                    "vscore": vscore,
                    "time_exe": time_exe,
                    "modphase": modphase_results,
                    "fidelity": fidelity,
                    "S_renyi": S_renyi,
                    "m": m,
                    "ms": ms,
                    "m2": m2,
                    "ms2": ms2,
                    "m_ED": m_ED,
                    "ms_ED": ms_ED,
                    "m2_ED": m2_ED,
                    "ms2_ED": ms2_ED,
                },
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
            )
