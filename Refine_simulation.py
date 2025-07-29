#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from netket.operator.spin import sigmaz
import optax
import json
import argparse
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
    save_results, dump_callback, init_model, architecture_label,
    get_filenames_from_settings, load_vstate, BestIterKeeper,
    EnergyPlotter
)
from NN_module.NN_utils import (
    activation_dict, scheduler_initializer, 
    phase_stats_ED, phase_stats_vstate,
    modphase
)
from NN_module.ST_utils import compare_params, masked_optimizer


parser = argparse.ArgumentParser()
parser.add_argument("-a",'--artifact_path', type=str, required=True, help='Path al artefacto principal que recoge los resultados de la simulacion')
args=parser.parse_args()
path_artifact= args.artifact_path

with open(path_artifact,'r') as f:
    artifact = json.load(f)

##### FILENAMES STUFF ######
# Get filenames in base
kwargs={
    'size':artifact['lattice']['size'],
    **artifact['coupling_model'],
    **artifact['model_NN']['setup']
} 
sim_label, _, json_label, title_label_callback = get_filenames_from_settings('Oxalate', artifact["model_NN"]["name"], **kwargs)

# Refinement filenames
write_folder = os.path.dirname(path_artifact) + "/Refinements/"
if not os.path.exists(write_folder):
    os.makedirs(write_folder)

if os.path.isfile(write_folder + json_label + ".json"):
    i=1
    file_temp = json_label + f"_{i}.json"
    while(os.path.isfile(write_folder + file_temp)):
        i+=1
        file_temp = json_label + f"_{i}.json"

    sim_label+=f"_{i}"
    json_label+=f"_{i}"
############################

# Load refinement configuration

with open("refinement.json",'r') as f:
    config = json.load(f)

epochs = config['lr_schedule']['epochs']  
sweeps=config['lr_schedule']['sweeps']                        # Simulation settings
schedule = config['lr_schedule']
n_samples = artifact['sampler']['n_samples']
#dump_simulation = config['dump_sim_callback']


# Load vstate....
vstate=load_vstate(artifact)

# Rebuild hamiltonian
size = artifact['lattice']['size']
N = int(np.prod(size))
strength = artifact['coupling_model']['strength']
theta = artifact['coupling_model']['theta']
phi = artifact['coupling_model']['phi']
oxa=OxalateJKGamma(
    size, 
    [strength, theta, phi],
    **{
        'bc': artifact['lattice']['bc'],
        'order': artifact['lattice']['order']
    }
)
H = Runner(oxa.cm).build_hamiltonian()

# Get exact diag energy, in the case.
E_ED = None
if artifact['results']['E_ED'] is not None:
    E_ED = artifact['results']['E_ED']

# Create log and keeper
log = (
    nk.logging.RuntimeLog()
)
keeper = BestIterKeeper(H, N, 1e-8)
plotter = EnergyPlotter(H)

# Learning Rate Schedule
transformations = {
    'train': optax.sgd(0.1),
    'freeze': optax.set_to_zero()
}

callback_artifacts = {}
time_in = time.time()


print(f"Epochs: {epochs}  Sweeps: {sweeps}")
            
if sweeps !=0:    # Alternated training between modulus and phase
    schedule_name = 'stairs_schedule'
    schedule = schedule[schedule_name]
    ds_schedule = jnp.linspace(1e-2, 1e-4, sweeps)
    lr_schedule = jnp.logspace(
        start=jnp.log10(schedule['lr0']),
        stop=jnp.log10(schedule['lr_min']), 
        num=sweeps
    )
    epochs_per_run = epochs//(2*sweeps)
    print(f"Epochs per run: {epochs_per_run}")

    for i in range(sweeps):

        print(f"\nSweep {i+1} of {sweeps}......   lr: {lr_schedule[i]:.4f}  ds: {ds_schedule[i]:.4f}\n")
        transformations['train'] = optax.sgd(learning_rate=lr_schedule[i])
        SR = nk.optimizer.SR(diag_shift=ds_schedule[i])

        for mask in ['modulus', 'phase']:
            mode = [m for m in ['phase', 'modulus'] if m != mask][0]
            
            variables = vstate.variables
            sampler = vstate.sampler
            optimizer = masked_optimizer(vstate.parameters, transformations, mode = mask)

            gs = nk.driver.VMC(
                H,
                optimizer,
                variational_state=vstate,
                preconditioner=SR
            )
            # print(jax.tree_util.tree_structure(vstate.parameters))
            # print(vstate.variables['params'].keys(  ))

            print(f"\nTraining {mode} for {epochs_per_run} epochs...")
            gs.run(n_iter=epochs_per_run, out=log, callback=[keeper.update, plotter], show_progress=True)
            mean, std, psi  = phase_stats_vstate(vstate)
            print(f"VS phase: {mean} \u00b1 {std}  ({psi})")


else:   # Training modulus and phase at the same time

    schedule_name = schedule['name']
    schedule = schedule[schedule_name]
    ds_schedule = optax.linear_schedule(1e-2, 1e-4, epochs)
    SR = nk.optimizer.SR(diag_shift=ds_schedule)
    lr_schedule = scheduler_initializer(schedule_name, schedule)
    optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)

    gs = nk.driver.VMC(
        H,
        optimizer,
        variational_state=vstate,
        preconditioner=SR
    ).run(n_iter=epochs, out=log, callback=[keeper.update, plotter], show_progress=True)

time_out = time.time()
time_exe= time_out - time_in

resp = input("Do you want to save simulation? [y/N]: ").strip().lower()
if resp not in ("y", "s", "si", "yes"):
    print("Execution stopped")
    sys.exit(0)
print("Saving....")




