#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import json
import time
import argparse
import system as sys
import glob
import ast

from VA_project.model.model import OxalateJKGamma
from VA_project.engine.runners import Runner
from NN_module.NN_utils import (
    dump_callback, save_results, BestIterKeeper,
    activation_dict, sampler_dict, optimizer_name_dict
)
from NN_module.models.MLP import (
    MultiLayerPerceptron, MultiLayerPerceptron_Z2_Traslation,
    MultiLayerPerceptron_Traslation
    )

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()

parser = argparse.ArgumentParser()
parser.add_argument("-d","--directory", type=str, required=True, help='Absolute path to the directory of results.json')
directory = parser.parse_args().directory

strength_list = []
theta_list = []
phi_list = []
sizes = []
kwargs_lattice_list = []
alpha_list = []

dimensions_list = []                   # MLP architecture settings
activation_list = []
opt_name_list = []
learning_rate_list = []
sampler_name_list = []
n_samples_list = []

if not directory.endswith('/'):
    directory += '/'

artifacts = glob.glob(directory + "**/*results*.json", recursive=True)
for file in artifacts:
    with open(file, 'r') as f:
        artifact = json.load(f)

    strength_list.append(artifact["coupling_model"]['strength'])                         # Lattice and coupling model
    theta_list.append(artifact["coupling_model"]['theta_list'])
    phi_list.append(artifact["coupling_model"]['phi_list'])
    sizes.append(artifact["lattice"]['sizes'])
    kwargs_lattice_list.append({'bc' : artifact["lattice"]['bc'], "order": "default_2"})
    n=int(np.prod(artifact['sizes'][0])) 

    dimensions_list.append(ast.literal_eval(artifact["model_NN"]['dense_dim']))            # MLP architecture settings
    alpha_list.append(tuple(dimensions_list[-1][0]//n, dimensions_list[-1][1]//n))  
    activation_list.append(ast.literal_eval(artifact["model_NN"]['activation']))
    opt_name_list.append(artifact['optimizer'])
    learning_rate_list.append(artifact['learning_rate'])
    
    sampler_name_list.append(artifact['sampler']['name'])            # Sampler settings
    n_samples_list.append(artifact['sampler']['n_samples'] ) 


with open("/home/ihuarte/Escritorio/Ivan/NNs/config.json",'r') as f:
    config = json.load(f)

iterations = config['iterations']                   # Simulation settings
exact_diag = config['exact_diagonalization']
dump_simulation = config['dump_sim_callback']
write = config['write_folder_sim']

diag_shift= 0.001       # Diagonal shift for the SR preconditioner

# alpha_list = [tuple(dim) for dim in alpha_list]
# activation_list = [[tuple(act) for act in activation] for activation in activation_list]

rng = jax.random.PRNGKey(666)


for i, (size,strength,theta,phi,kwargs_lattice, alphas, activation_name, 
        opt_name, learning_rate, sampler_name, n_samples,dimensions) in enumerate(
            zip(sizes, strength_list, theta_list, phi_list, kwargs_lattice_list, 
                alpha_list, activation_list, opt_name_list, learning_rate_list, 
                sampler_name_list, n_samples_list, dimensions_list)
        ):
    
    activation = tuple([activation_dict[act] if act != 0 else 0 for act in activation_name])

    N = int(np.prod(size)) 
    if N > 20: exact_diag = False

    write_folder = write +  f"Oxalate_size_{size[0]}x{size[1]}/"

    # Create the Hamiltonian
    oxa=OxalateJKGamma(
        size, 
        [strength, theta, phi],
        **kwargs_lattice
    )
    H = Runner(oxa.cm).build_hamiltonian()

    if exact_diag:
        E_ED, x_ED = Runner(oxa.cm).exact_energy_lanczos(eigenstates=True)
        E_ED = float(E_ED.squeeze(-1))

    print(f"\n---- Parameters: Size {size}  strength={strength:1f}  theta={theta:2f}  phi={phi:2f} ----\n\n")
    print(f"alpha: {alphas} \nactivation: {activation_name} \nopt_name: {opt_name} \nlearning_rate: {learning_rate}")

    callback_artifacts = {}
    time_in = time.time()
    
    # Initialize the model
    model = MultiLayerPerceptron(
            N = N,
            param_dtype=jnp.complex128,
            hidden_alpha=alphas,
            activation=activation,
        )
    #params = model.init(rng,jnp.ones((1,16), dtype=jnp.complex64))

    hi = nk.hilbert.Spin(s=0.5, N = int(np.prod(size)))

    sampler = sampler_dict[sampler_name](hi, dtype=complex)

    optimizer = optimizer_name_dict[opt_name](learning_rate=learning_rate)

    vstate = nk.vqs.MCState(sampler, model, n_samples = n_samples)
    is_holo = nk.utils.is_probably_holomorphic(vstate._apply_fun, vstate.parameters, vstate.samples, vstate.model_state)

    SR = nk.optimizer.SR(diag_shift = diag_shift, holomorphic = True)

    gs = nk.VMC(
        hamiltonian=H,
        optimizer=optimizer,
        preconditioner=SR,
        variational_state=vstate)
    
    log = nk.logging.RuntimeLog()
    keeper= BestIterKeeper(H, np.prod(size), baseline = 1e-8,
        #filename=pathlib.Path("best_state.nk"),
    )
    gs.run(n_iter=iterations, out=log, callback= [keeper.update])
    time_out = time.time()
    time_exe= time_out - time_in
    
    if exact_diag:
        keeper.E_ED = E_ED
        log.E_ED = E_ED


    sim_label = f"_{opt_name}_{learning_rate}_sim_{i}"
    if dump_simulation:
        # For plotting architecture
        set_dim=[f"D({dim})" for dim in dimensions] 
        set_act=[f"A({act})" for act in activation_name]
        setup='||'
        for i in range(len(dimensions)):
            setup+=f" {set_dim[i]} |"
            setup+=f" {set_act[i]} |" if '0' not in set_act[i] else ''
            
        setup+="|"
        
        dump_setup ={
                    'size': size, 'theta': theta,
                    'phi': phi, 'opt_name': opt_name,
                    'learning_rate': learning_rate,
                    'write_folder': write_folder,
                    'time_exe': time_exe, 'architecture': setup,
                    'sim_label': sim_label
                }
        
        callback_artifacts = dump_callback(log, dump_setup, write=True)

    else:
        callback_artifacts = None
        

    # Extract some results
    vstate = keeper.best_state
    E_best = float(keeper.best_energy)
    vscore = float(keeper.vscore)

    if exact_diag:
        error=float(np.abs(E_best-E_ED)/np.abs(E_ED))
    else:
        E_ED = None
        x_ED = None
        error = None    

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
            'name': 'MLP',
            'alphas' : alphas,
            'activation': activation_name
            
        },
        'sampler': {
            'name': sampler_name,
            'n_samples': n_samples, 
            'rng': vstate.sampler_state.rng.tolist() 

        },
        'optimizer': opt_name,
        'learning_rate': learning_rate,

        'results':{
            'E_best': E_best,
            'E_ED': E_ED,
            'error': error,
            'vscore': vscore,
            'time_exe': time_exe, 

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
        sim_label = sim_label
    )
    
