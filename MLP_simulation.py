#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import json
import optax
import time

# Añadir los directorios necesarios
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA/VA_project/src"))


from VA_project.model.model import OxalateJKGamma
from VA_project.engine.runners import Runner
from NN_module.NN_utils import (
    activation_dict, sampler_dict, scheduler_initializer, 
    phase_stats_ED, phase_stats_vstate
)
from NN_module.sim_utils import (
    save_results, dump_callback, BestIterKeeper
)
from NN_module.models.MLP import BatchedMultiLayerPerceptron

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()

with open("/home/ihuarte/Escritorio/Ivan/NNs/config.json",'r') as f:
    config = json.load(f)

sizes = config['sizes']
strength = config['strength']
theta_list = config['theta_list']
phi_list = config['phi_list']
kwargs_lattice = config['kwargs_lattice']

alpha_list = config['alpha_list']         # MLP architecture settings
activation_list = config['activation_list']
symm_2D = config['symm_2D']
symm_Z2 = config['symm_Z2']
trivial_Z2 = config['trivial_Z2']

print(f"symm_2D: {symm_2D}")
print(f"symm_Z2: {symm_Z2}")
print(f"trivial_Z2: {trivial_Z2}\n \n")

sampler_name = config['sampler']['name']            # Sampler settings
n_samples = config['sampler']['n_samples']

epochs = config['lr_schedule']['epochs']                   # Simulation settings
schedule = config['lr_schedule']
exact_diag = config['exact_diagonalization']
dump_simulation = config['dump_sim_callback']
write = config['write_folder_sim']

### Training schedule ###

lr_schedule = scheduler_initializer(schedule['name'], schedule)
optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
ds_schedule = optax.linear_schedule(1e-2, 1e-4, epochs)  # Diagonal shift for the SR preconditioner
SR = nk.optimizer.SR(diag_shift=ds_schedule, holomorphic = True)

alpha_list = [tuple(dim) for dim in alpha_list]
activation_list = [[tuple(act) for act in activation] for activation in activation_list]

rng = jax.random.PRNGKey(666)

alpha_list = [tuple(dim) for dim in alpha_list]
activation_list = [[tuple(act) for act in activation] for activation in activation_list]

rng = jax.random.PRNGKey(666)

d=0 ; a=0 
for i, size in enumerate(sizes):

    N = int(np.prod(size)) 
    if N > 20: 
        exact_diag = False
    else:
        exact_diag = True

    hi = nk.hilbert.Spin(s=0.5, N = int(np.prod(size)))
    sampler = sampler_dict[sampler_name](hi, dtype=complex)

    write_folder = write +  f"Oxalate_size_{size[0]}x{size[1]}/"

    for j, (theta, phi) in enumerate(zip(theta_list, phi_list)):

        # Update Hamiltonian
        oxa=OxalateJKGamma(
            size, 
            [strength, theta, phi],
            **kwargs_lattice
        )
        H = Runner(oxa.cm).build_hamiltonian()

        if exact_diag:# and not os.path.isfile(write_folder + f"Oxalate_xED_{size[0]}x{size[1]}_strength_{strength:.1f}_theta_{theta:.1f}_phi_{phi:.1f}.txt"):
            print("Running exact diagonalization...")
            E_ED, x_ED = Runner(cm).exact_energy_lanczos(eigenstates=True)
            E_ED = float(E_ED.squeeze(-1))
            print(f"Energy ED: {E_ED}")

        for a,alphas in enumerate(alpha_list): 

            dimensions = tuple([int(a*N) for a in alphas])
            
            for activation_name in activation_list[a]:

                activation = tuple([activation_dict[act] if act != 0 else 0 for act in activation_name])
                
                callback_artifacts = {}
                time_in = time.time()
                
                # Initialize the model
                model = BatchedMultiLayerPerceptron(
                        lattice_size=tuple(size),
                        param_dtype=jnp.complex128,
                        hidden_alpha=alphas,
                        activation=activation,
                        symm_2D=symm_2D,
                        symm_Z2=symm_Z2,
                        trivial_Z2=trivial_Z2
                    )
            
                vstate = nk.vqs.MCState(sampler, model, n_samples = n_samples)

                gs = nk.VMC(
                    hamiltonian=H,
                    optimizer=optimizer,
                    preconditioner=SR,
                    variational_state=vstate)
                
                log = nk.logging.RuntimeLog()
                keeper= BestIterKeeper(H, np.prod(size), baseline = 1e-8,
                    #filename=pathlib.Path("best_state.nk"),
                )
                gs.run(n_iter=epochs, out=log, callback= [keeper.update])
                time_out = time.time()
                time_exe= time_out - time_in
                
                if exact_diag:
                    keeper.E_ED = E_ED
                    log.E_ED = E_ED

                ## Save Callback
                title_label='MLP Oxalate'
                act_label=''
                dim_label=''
                for act in activation_name:
                    act_label += f"{act}_"
                for a in alphas:
                    dim_label += f"{a}_"

                coup_label = f"Oxalate_simulation_{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}"
                nn_label = f"alphas_{dim_label}_activations_{act_label}"
                title_label += f"a: {strength} " + r"$\theta:%.1f \;\; \phi: %.1f$"%(theta, phi)

                sim_label = f"Oxalate_simulation_" + coup_label + nn_label 
                ED_label = f"Oxalate_xED_" + coup_label
                json_label = f"Oxalate_results_" + nn_label
                title_label_callback = f"Callback  "+ title_label + f"  ({size[0]}x{size[1]})"            

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
                            'size': size, 'opt_name': "Sgd",
                            'learning_rate': "Scheduled",
                            'write_folder': write_folder,
                            'time_exe': time_exe, 
                            'architecture': setup,
                            'sim_label': sim_label,
                            'title_label_callback': title_label_callback
                        }
                    
                    callback_artifacts = dump_callback(log, dump_setup, write=False)

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
                        'name': 'MLP',
                        "lattice_size":size,
                        "param_dtype":"jnp.complex128",
                        'hidden_alpha': alphas,
                        'activation': activation_name,
                        "symm_2D":symm_2D,
                        "symm_Z2":symm_Z2,
                        "trivial_Z2":trivial_Z2
                        
                    },
                    'sampler': {
                        'name': sampler_name,
                        'n_samples': n_samples, 
                        'rng': vstate.sampler_state.rng.tolist() 

                    },
                    'optimizer': 'Sgd',
                    
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
                        'phase': phase

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
                
