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
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()

# Añadir los directorios necesarios
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA/VA_project/src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "Transformers/transformer_LR_WF_public"))

# Importar módulos necesarios
from VA_project.model.model import OxalateJKGamma
from VA_project.engine.runners import Runner
from NN_module.sim_utils import (
    save_results, dump_callback, init_model
)
from NN_module.label_utils import (
    get_filenames_from_settings, architecture_label,
    get_write_folder_from_model, display_simulation_settings
)
from NN_module.NN_utils import (
    scheduler_initializer, 
    phase_stats_vstate,
    modphase
)
from NN_module.ST_utils import compare_params, masked_optimizer
from transformer_LR_WF.utils import *

# Cargamos configuracion de archivo json
with open("config_split_training.json",'r') as f:
    config = json.load(f)

cm_model_name = config['CM']['selection']
nn_model_name=config["model_NN"]["selection"]
nn_model_setup = config["model_NN"][nn_model_name]
model_label= cm_model_name+'_'+nn_model_name

sizes = config['sizes']
strength = config['CM'][cm_model_name]['strength']                # Lattice and coupling model
theta_list = config['CM'][cm_model_name]['theta_list']
phi_list = config['CM'][cm_model_name]['phi_list']
kwargs_lattice = config['kwargs_lattice']

epochs = config['lr_schedule']['epochs']                          # Simulation settings
schedule = config['lr_schedule']
exact_diag = config['exact_diagonalization']
dump_simulation = config['dump_sim_callback']

sampler_setup = config['sampler']
n_samples = sampler_setup['n_samples_per_chain'] * sampler_setup['n_chains_per_rank'] * sampler_setup['n_ranks']
print(f"Total samples: {n_samples}")

write = get_write_folder_from_model(config)

### MC sampling rules ###
rule1 = nk.sampler.rules.LocalRule()
rule2 = InvertMagnetization()
pinvert = 0.25
pflip = 1 - pinvert


E_ED = None
x_ED = None

for i, size in enumerate(sizes):
    nn_model_setup['lattice_size']=size

    N = int(np.prod(size)) 
    if N > 20: exact_diag = False

    write_folder_size = write +  f"Size_{size[0]}x{size[1]}/"

    ###  Reseting Hilbert space object and the observables ###
    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    renyi = nkx.observable.Renyi2EntanglementEntropy(
        hi, np.arange(0, N / 2 + 1, dtype=int)
    )
    mags = sum([(-1) ** (i+j) * sigmaz(hi, i*size[1]+j) / N for i in range(size[0]) for j in range(size[1])])
    magnet = sum([sigmaz(hi, i*size[1]+j) / N for i in range(size[0]) for j in range(size[1])])

    ## Reset sampler 
    sampler = nk.sampler.MetropolisSampler(
    hi, nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert]),
    n_chains_per_rank=sampler_setup['n_chains_per_rank'], chunk_size=sampler_setup['chunk_sampler'],
    )

    for j, (theta, phi) in enumerate(zip(theta_list, phi_list)):

        config['CM'][cm_model_name]['theta']=theta
        config['CM'][cm_model_name]['phi']=phi
        config['size']=size
        display_simulation_settings(config)
        
        ## Update Hamiltonian
        oxa=OxalateJKGamma(
            size, 
            [strength, theta, phi],
            **kwargs_lattice
        )
        H = Runner(oxa.cm).build_hamiltonian()

        if exact_diag:
            print("Running exact diagonalization...")
            E_ED, x_ED = Runner(oxa.cm).exact_energy_lanczos(eigenstates=True)
            E_ED = float(E_ED.squeeze(-1))
            print(f"Energy ED: {E_ED}")

        for sweeps in schedule['sweeps']:
            
            print(f"\nRunning with {sweeps} sweeps...")
            write_folder = write_folder_size + f"sweeps_{sweeps}/"
            if sweeps !=0:
                schedule_name = 'stairs_schedule'
                schedule_selection = config['lr_schedule'][schedule_name]

                schedule_selection['sweeps'] = sweeps

                ds_schedule = jnp.linspace(1e-2, 1e-4, sweeps)
                lr_schedule = jnp.logspace(
                    start=jnp.log10(schedule_selection['lr0']),
                    stop=jnp.log10(schedule_selection['lr_min']), 
                    num=sweeps
                )

            else:
                schedule_name = config['lr_schedule']['name']
                schedule_selection = config['lr_schedule'][schedule_name]

                ds_schedule = optax.linear_schedule(1e-2, 1e-4, epochs)
                SR = nk.optimizer.SR(diag_shift=ds_schedule)
                lr_schedule=lr_schedule = scheduler_initializer("warmup_exponential_decay", config['lr_schedule'])
                optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
            
            callback_artifacts = {}
            time_in = time.time()

            model = init_model(nn_model_name, nn_model_setup)

            log = (
                nk.logging.RuntimeLog()
            )  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.
            keeper = BestIterKeeper(H, N, 1e-8)
            # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

            # Initialize vstate with parameters
            vstate = nk.vqs.MCState(
                sampler, model=model, n_samples=n_samples,
                n_discard_per_chain=0, chunk_size=sampler_setup['chunk_vstate']
            )

            print(f"Epochs: {epochs}  Sweeps: {sweeps}")
            
            if sweeps !=0:    # Alternated training between modulus and phase

                epochs_per_run = epochs//(2*sweeps)
                print(f"Epochs per run: {epochs_per_run}")

                transformations = {
                    'train': optax.sgd(0.1),
                    'freeze': optax.set_to_zero()
                }

                for i in range(sweeps):

                    print(f"\nSweep {i+1} of {sweeps}......   lr: {lr_schedule[i]:.4f}  ds: {ds_schedule[i]:.4f}\n")
                    transformations['train'] = optax.sgd(learning_rate=lr_schedule[i])
                    SR = nk.optimizer.SR(diag_shift=ds_schedule[i])

                    for mask in ['modulus', 'phase']:
                        mode = [m for m in ['phase', 'modulus'] if m != mask][0]
                        
                        variables = vstate.variables
                        sampler = vstate.sampler
                        optimizer = masked_optimizer(vstate.parameters, transformations, mode = mask)
                        
                        vstate = nk.vqs.MCState(
                            sampler,
                            sampler_seed=vstate.sampler_state.rng,
                            model=model,
                            n_samples=n_samples,
                            n_discard_per_chain=0,
                            chunk_size=sampler_setup['chunk_vstate'],
                            variables=variables
                        )
    
                        gs = nk.driver.VMC(
                            H,
                            optimizer,
                            variational_state=vstate,
                            preconditioner=SR
                        )
                        # print(jax.tree_util.tree_structure(vstate.parameters))
                        # print(vstate.variables['params'].keys(  ))


                        print(f"\nTraining {mode} for {epochs_per_run} epochs...")
                        gs.run(n_iter=epochs_per_run, out=log, callback=[keeper.update], show_progress=True)
                        mean, std, psi  = phase_stats_vstate(vstate)
                        print(f"VS phase: {mean} \u00b1 {std}  ({psi})")


            else:       # Training modulus and phase at the same time
                gs = nk.driver.VMC(
                    H,
                    optimizer,
                    variational_state=vstate,
                    preconditioner=SR
                ).run(n_iter=epochs, out=log, callback=[keeper.update], show_progress=True)

            time_out = time.time()
            time_exe= time_out - time_in
            
            if exact_diag:
                keeper.E_ED = E_ED
                log.E_ED = E_ED

            ## Save results
            _kwargs = config['model_NN'][nn_model_name]
            _kwargs['size']=size ; _kwargs['strength']=strength 
            _kwargs['theta']=theta ; _kwargs['phi']=phi
            
            sim_label, ED_label, json_label, title_label_callback = get_filenames_from_settings(config['CM']['selection'], config['model_NN']['selection'], **_kwargs )

            if dump_simulation:
                # For plotting architecture
                architecture=architecture_label(nn_model_name, nn_model_setup)
                dump_setup ={
                    'size': size, 'opt_name': "Sgd",
                    'learning_rate': "Scheduled",
                    'write_folder': write_folder,
                    'time_exe': time_exe, 
                    'architecture': architecture,
                    'sim_label': sim_label,
                    'title_label_callback': title_label_callback
                }
                
                callback_artifacts = dump_callback(log, dump_setup)

            else:
                callback_artifacts = None

            ## Calculate some observables
            # Modulus and phase

            vstate = keeper.best_state
            E_best = float(keeper.best_energy)
            vscore = float(keeper.vscore)

            modphase_results={}
            if exact_diag:
                error=float(np.abs(E_best-E_ED)/np.abs(E_ED))
                mp_array_ED, stats_ED = modphase(x_ED)
                modphase_results['xED']= stats_ED
                print(f"xED phase: {stats_ED['phase']['mean']} \u00b1 {stats_ED['phase']['std']}  ({stats_ED['type']})")

            else:
                E_ED = None
                x_ED = None
                error = None

            mp_array_vs, stats_vs = modphase(vstate)
            modphase_results['vstate']= stats_vs
            print(f"vstate phase: {stats_vs['phase']['mean']} \u00b1 {stats_vs['phase']['std']}  ({stats_vs['type']})")

            # Fidelity
            fidelity = float(jnp.abs(jnp.vdot(vstate.to_array(), x_ED.squeeze())))
            print(f"Fidelity: {fidelity:.3e}")

            # Renyi entropy, magnetization and its fluctuation
            S_renyi = float(vstate.expect(renyi).mean)
            M = float(vstate.expect(magnet).mean.real)
            Ms = float(vstate.expect(mags).mean.real)

            print(f"Renyi entropy: {S_renyi}")
            print(f"Magnetization: {M}")
            print(f"Magnetization fluctuation: {Ms}")
            ## Save the results

            dump_setup ={
                'model_label': model_label,

                'lattice':{
                    'name': 'Triangular',
                    'size': size, 
                    'bc': kwargs_lattice['bc'],
                    'order': kwargs_lattice['order']
                },

                'coupling_model': {
                    'strength': strength,
                    'theta': theta,
                    'phi': phi, 
                },

                'model_NN': {
                    "name": nn_model_name,
                    "setup":nn_model_setup,
                },

                'sampler': {
                    'name': "MetropolisSampler",
                    'n_samples': n_samples, 
                    'rng': vstate.sampler_state.rng.tolist(), 
                    'rules': 'LocalRule/InvertMagnetization',
                    "setup": sampler_setup

                },
                'optimizer': "Sgd",
                "lr_schedule":{
                    "name": schedule["name"],
                    "setup": schedule[schedule["name"]]
                },

                'results':{
                    'E_best': E_best,
                    'E_ED': E_ED,
                    'error': error,
                    'vscore': vscore,
                    'time_exe': time_exe,
                    'modphase': modphase_results,
                    "fidelity": fidelity,
                    'S_renyi': S_renyi,
                    'M': M,
                    'Ms': Ms
                    
                },
                '_artifacts': {
                    'callback': callback_artifacts
                }
            }

            save_results(
                vstate, 
                dump_setup, 
                x_ED = x_ED,
                modphase=mp_array_vs,
                modphase_ED= mp_array_ED,
                write_folder = write_folder,
                sim_label = sim_label,
                ED_label=ED_label,
                json_label=json_label
            )

            # import time
            # import subprocess
            # time.sleep(2)

            # artifact_path = write_folder + json_label + ".json"
            # script_path = "/home/ihuarte/Escritorio/Ivan/NNs/plot_phase.py"

            # subprocess.run(["python", script_path, "-a", artifact_path])








        

