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
    save_results, dump_callback, init_model, architecture_label
)
from NN_module.NN_utils import (
    activation_dict, scheduler_initializer, 
    phase_stats_ED, phase_stats_vstate
)
from NN_module.ST_utils import compare_params, masked_optimizer
from transformer_LR_WF.utils import *

# Cargamos configuracion de archivo json

with open("config_split_training.json",'r') as f:
    config = json.load(f)

cm_name = config['CM']['selection']
strength = config['CM'][cm_name]['strength']                # Lattice and coupling model
theta_list = config['CM'][cm_name]['theta_list']
phi_list = config['CM'][cm_name]['phi_list']
sizes = config['sizes']
kwargs_lattice = config['kwargs_lattice']

model_name=config["model_NN"]["selection"]
model_setup = config["model_NN"][model_name]
model_label= cm_name+'_'+model_name

token_size=model_setup['token_size']              # ViT architecture settings
embedding_d=model_setup['embedding_d']
n_heads=model_setup['n_heads']
n_blocks= model_setup['n_blocks']
n_ffn_layers = model_setup['n_ffn_layers']
final_architecture = ast.literal_eval(model_setup['final_architecture'] )
is_complex = model_setup['is_complex']
symm_2D_module = model_setup['symm_2D_module']
symm_Z2_module = model_setup['symm_Z2_module']
trivial_Z2_module = model_setup['trivial_Z2_module']

# alpha_list = model_setup['alpha_list']            # MLP architecture settings
# activation_list = model_setup['activation_list']
# symm_2D_phase = model_setup['symm_2D_phase']
# symm_Z2_phase = model_setup['symm_Z2_phase']
# trivial_Z2_phase = model_setup['trivial_Z2_phase']
# output_dim = model_setup['output_dimension']
# alphas = alpha_list[0]

epochs = config['lr_schedule']['epochs']        # Simulation settings
schedule=config['lr_schedule']
stairs = config['lr_schedule'][config['lr_schedule']['name']]
n_samples = config['n_samples']
exact_diag = config['exact_diagonalization']
dump_simulation = config['dump_sim_callback']

if  model_name=="SplitTraining_ViT_MLP": 
    conditions = [symm_2D_module, symm_2D_phase,symm_Z2_module,symm_Z2_phase],
    labels = ["_2DM","_2DP","_Z2M","_Z2P"]
    print(f"symm_2D_module: {symm_2D_module}     symm_2D_phase: {symm_2D_phase}")
    print(f"symm_Z2_module: {symm_Z2_module}     symm_Z2_phase: {symm_Z2_phase}")
    print(f"trivial_Z2_module: {trivial_Z2_module} trivial_Z2_phase: {trivial_Z2_phase}")

elif  model_name == "SplitTraining_ViT_CNN":
    conditions = [symm_2D_module,symm_Z2_module],
    labels = ["_2DM","_Z2M"]
    print(f"symm_2D_module: {symm_2D_module}    symm_Z2_module: {symm_Z2_module}    trivial_Z2_module: {trivial_Z2_module}")


symm=''
for cond, label in zip():
    if cond:
        symm+=label
        if label == '_Z2M':  symm += "t" if trivial_Z2_module else "nt"
        if label == '_Z2P':  symm += "t" if trivial_Z2_phase else "nt"
    
write = config['write_folder_sim'] + model_label+ symm + "/"


### MC sampling rules ###
rule1 = nk.sampler.rules.LocalRule()
rule2 = InvertMagnetization()
pinvert = 0.25
pflip = 1 - pinvert



### Training schedule ###
lr_schedule = jnp.logspace(
    start=jnp.log10(stairs['lr0']),
    stop=jnp.log10(stairs['lr_min']), 
    num=stairs['sweeps']
)

transformations = {
    'train': optax.sgd(0.1),
    'freeze': optax.set_to_zero()
}

#lr_schedule=scheduler_initializer(schedule['name'], schedule)
ds_schedule = jnp.linspace(1e-1, 1e-4, stairs['sweeps'])

E_ED = None
x_ED = None

for i, size in enumerate(sizes):
    model_setup['lattice_size']=size

    N = int(np.prod(size)) 
    if N > 20: exact_diag = False

    write_folder_size = write +  f"Size_{size[0]}x{size[1]}/"

    ###  Reseting Hilbert space object and the observables ###
    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    renyi = nkx.observable.Renyi2EntanglementEntropy(
        hi, np.arange(0, N / 2 + 1, dtype=int)
    )
    mags = sum([(-1) ** i * sigmaz(hi, i) / N for i in range(N)])
    magnet = sum([sigmaz(hi, i) / N for i in range(N)])

    ## Reset sampler 
    sampler = nk.sampler.MetropolisSampler(
    hi, nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert])
    )

    for j, (theta, phi) in enumerate(zip(theta_list, phi_list)):

        print(f"\n---- Parameters: Size {size}  strength={strength:1f}  theta={theta:2f}  phi={phi:2f} ----\n\n")
        
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

        for sweeps in [50]:
            stairs['sweeps'] = sweeps if stairs['sweeps'] != 0 else 0
            print(f"\nRunning with {stairs['sweeps']} sweeps...")
            write_folder = write_folder_size + f"sweeps_{sweeps}/"
            if stairs['sweeps'] !=0:
                ds_schedule = jnp.linspace(1e-2, 1e-4, stairs['sweeps'])
                lr_schedule = jnp.logspace(
                    start=jnp.log10(stairs['lr0']),
                    stop=jnp.log10(stairs['lr_min']), 
                    num=stairs['sweeps']
                )

            else:
                ds_schedule = optax.linear_schedule(1e-2, 1e-4, epochs)
                SR = nk.optimizer.SR(diag_shift=ds_schedule)
                lr_schedule=lr_schedule = scheduler_initializer("warmup_exponential_decay", config['lr_schedule'])
                optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
            

            callback_artifacts = {}
            time_in = time.time()

            model = init_model(model_name, model_setup)

            log = (
                nk.logging.RuntimeLog()
            )  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.
            keeper = BestIterKeeper(H, N, 1e-8)
            # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

            # Initialize vstate with parameters
            vstate = nk.vqs.MCState(
                sampler, model=model, n_samples=n_samples,
                n_discard_per_chain=0, chunk_size=None
            )
            #params0 = vstate.parameters



            print(f"Epochs: {epochs}  Sweeps: {stairs['sweeps']}")
            
            if stairs['sweeps'] !=0:    # Alternated training between modulus and phase

                epochs_per_run = epochs//(2*stairs['sweeps'])
                print(f"Epochs per run: {epochs_per_run}")

                for i in range(stairs['sweeps']):

                    print(f"\nSweep {i+1} of {stairs['sweeps']}......   lr: {lr_schedule[i]:.4f}  ds: {ds_schedule[i]:.4f}\n")
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
                            chunk_size=None,
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
                    

            vstate=keeper.best_state

            time_out = time.time()
            time_exe= time_out - time_in
            
            if exact_diag:
                keeper.E_ED = E_ED
                log.E_ED = E_ED

            ## Save results
            
            sim_label = f"Oxalate_simulation_"+model_label+f"{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}"
            ED_label = f"Oxalate_xED_{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_ED"
            json_label = f"Oxalate_results_"+model_name+f"_{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}"
            title_label_callback = f"Callback Split "+model_name+" " + r"$\theta = %.1f$  $\phi = %.1f$"%(theta,phi) + f"  ({size[0]}x{size[1]})"

            #sim_label, ED_label, json_label,title_label_callback =  get_filenames_from_settings()

            if dump_simulation:
                # For plotting architecture
                architecture=architecture_label(model_name, model_setup)
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
                

            # Extract some results
            vstate = keeper.best_state
            E_best = float(keeper.best_energy)
            vscore = float(keeper.vscore)

            phase={}
            if exact_diag:
                error=float(np.abs(E_best-E_ED)/np.abs(E_ED))
                mean_ED, std_ED, psi_ED = phase_stats_ED(x_ED)
                phase['xED']={'mean':mean_ED, 'std':std_ED, 'psi': psi_ED}
                print(f"xED phase: {mean_ED} \u00b1 {std_ED}  ({psi_ED})")

            else:
                E_ED = None
                x_ED = None
                error = None

            mean, std, psi  = phase_stats_vstate(vstate)
            phase['vstate']={'mean':mean, 'std':std, 'psi': psi}
            print(f"VS phase: {mean} \u00b1 {std}  ({psi})\n \n")

            # Save the results

            dump_setup ={

                'lattice':{
                    'name': 'Triangular',
                    'size': size, 
                    'bc': kwargs_lattice['bc'],
                },

                'coupling_model': {
                    'strength': strength,
                    'theta': theta,
                    'phi': phi, 
                },

                'model_NN': {
                    "name": model_name,
                    "setup":model_setup,
                },

                'sampler': {
                    'name': "MetropolisSampler",
                    'n_samples': n_samples, 
                    'rng': vstate.sampler_state.rng.tolist(), 
                    'rules': 'LocalRule/InvertMagnetization'

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
                    'time_exe': time_exe
                },
                '_artifacts': {
                    'callback': callback_artifacts
                }
            }

            save_results(
                vstate, 
                dump_setup, 
                x_ED = x_ED,
                write_folder = write_folder,
                sim_label = sim_label,
                ED_label=ED_label,
                json_label=json_label
            )

            if stairs['sweeps'] ==0:
                break

            # import time
            # import subprocess
            # time.sleep(2)

            # artifact_path = write_folder + json_label + ".json"
            # script_path = "/home/ihuarte/Escritorio/Ivan/NNs/plot_phase.py"

            # subprocess.run(["python", script_path, "-a", artifact_path])








        

