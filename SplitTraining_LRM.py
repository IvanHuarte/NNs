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
import argparse

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
from VA_project.model.model import LRChain, LRSquare
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
from NN_module.ST_utils import compare_params, masked_optimizer
from NN_module.observables import calc_all_observables_vs, calc_all_observables_ED
from transformer_LR_WF.utils import InvertMagnetization

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config",
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
J_list = config_cm["CM"][cm_model_name]["J_list"]  # Lattice and coupling model
alpha_list = config_cm["CM"][cm_model_name]["alpha_list"]
fields = config_cm["CM"][cm_model_name]["fields"]
operators = config_cm["CM"][cm_model_name]["ops"]
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

    for alpha in alpha_list:

        for J in J_list:

            config_cm["CM"][cm_model_name]["J"] = J
            config_cm["CM"][cm_model_name]["alpha"] = alpha
            config_cm["size"] = size
            display_simulation_settings({**config_cm, **config_nn})

            ## Update Hamiltonian
            if cm_model_name == "LRChain":
                lrm = LRChain(  # We divide J and X field by the size to match Sebas's hamiltonian except by a constant factor
                    size[0] * size[1],
                    J / size[0],
                    alpha,
                    [fields[0] / size[0], fields[1] / size[0]],
                    ops=operators,
                    **kwargs_lattice,
                )
            elif cm_model_name == "LRSquare":
                lrm = LRSquare(  # We divide J and X field by the size to match Sebas's hamiltonian except by a constant factor
                    size,
                    J / (size[0] * size[1]),
                    alpha,
                    [fields[0] / (size[0] * size[1]), fields[1] / (size[0] * size[1])],
                    ops=operators,
                    **kwargs_lattice,
                )
            eng = Runner(lrm.cm, S_operators=False)
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
                ds_schedule = jnp.linspace(1e-2, 1e-4, total_segments)

                transformations = {
                    "train": optax.sgd(0.1),
                    "freeze": optax.set_to_zero(),
                }

                keeper = BestIterKeeper(
                    total_epochs, H, N, baseline=1e-8, mode="best_vscore"
                )
                # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

                print(f"Epochs:      {total_epochs}")
                print(f"Segments:    {segments}")
                print(f"Modes:       {modes}")

                for i, (segment, seg_modes, lr) in enumerate(
                    zip(segments, modes, lr_schedule)
                ):

                    SR = nk.optimizer.SR(diag_shift=ds_schedule[i])
                    print(
                        f"\nSegment {i+1} of {total_segments}......   lr: {lr:.4f}  ds: {ds_schedule[i]:.4f}\n"
                    )

                    for epochs, mode in zip(segment, seg_modes):
                        # P0 = vstate.parameters
                        # vs_0 = vstate

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
                        optimizer, _ = masked_optimizer(
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
                        # compare_params(P0, P1)
                        # check_zero_grads(vstate, mask)
                        # vs_1 = vstate

            else:  # Training modulus and phase at the same time
                keeper = BestIterKeeper(epochs, H, N, baseline=1e-8, mode="best_energy")
                # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

                training_setup["total_epochs"] = total_epochs

                ds_schedule = optax.linear_schedule(1e-2, 1e-4, total_epochs)
                SR = nk.optimizer.SR(diag_shift=ds_schedule)
                lr_schedule = scheduler_initializer(
                    "warmup_exponential_decay", lr_schedule_setup
                )
                optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
                gs = nk.driver.VMC(
                    H, optimizer, variational_state=vstate, preconditioner=SR
                ).run(
                    n_iter=epochs,
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
            _kwargs["J"] = J
            _kwargs["alpha"] = alpha
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
                    "J": J,
                    "alpha": alpha,
                    "fields": fields,
                    "ops": operators,
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
