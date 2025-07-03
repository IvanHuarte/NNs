#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import netket as nk
import netket.experimental as nkx
from netket.operator.spin import sigmaz
import optax
import json
import time
import ast
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()

# Añadir los directorios necesarios
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ATMOS_VA/VA_project/src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "Transformers/transformer_LR_WF_public"))

# Importar módulos necesarios
from VA_project.lattice.lattice import Chain
from VA_project.model.cm import GeneralNeighborCoupling
from VA_project.engine.runners import Runner
from NN_module.sim_utils import (
    save_results, dump_callback
)
from NN_module.NN_utils import (
    activation_dict, scheduler_initializer, 
    phase_stats_ED, phase_stats_vstate
)
from transformer_LR_WF.utils import *
from NN_module.models.split_training import SplitTraining_ViT_MLP

# Cargamos configuracion de archivo json

with open("/home/ihuarte/Escritorio/Ivan/NNs/config_split_training.json",'r') as f:
    config = json.load(f)

strength = config['strength']               # Lattice and coupling model
theta_list = config['theta_list']
phi_list = config['phi_list']
sizes = config['sizes']
kwargs_lattice = config['kwargs_lattice']

token_size=config['token_size']              # ViT architecture settings
embedding_d=config['embedding_d']
n_heads=config['n_heads']
n_blocks= config['n_blocks']
n_ffn_layers = config['n_ffn_layers']
final_architecture = ast.literal_eval(config['final_architecture'] )
is_complex = config['is_complex']
symm_2D_module = config['symm_2D_module']
symm_Z2_module = config['symm_Z2_module']
trivial_Z2_module = config['trivial_Z2_module']

alpha_list = config['alpha_list']             # MLP architecture settings
activation_list = config['activation_list']
symm_2D_phase = config['symm_2D_phase']
symm_Z2_phase = config['symm_Z2_phase']
trivial_Z2_phase = config['trivial_Z2_phase']
output_dim = config['output_dimension']
alphas = alpha_list[0]
activation_name = activation_list[0]
activations = [activation_dict[a] for a in activation_name]

epochs = config['lr_schedule']['epochs']                   # Simulation settings
schedule=config['lr_schedule']
n_samples = config['n_samples']
exact_diag = config['exact_diagonalization']
dump_simulation = config['dump_sim_callback']
write = config['write_folder_sim']


### MC sampling rules ###
rule1 = nk.sampler.rules.LocalRule()
rule2 = InvertMagnetization()
pinvert = 0.25
pflip = 1 - pinvert


print(f"symm_2D_module: {symm_2D_module}     symm_2D_phase: {symm_2D_phase}")
print(f"symm_Z2_module: {symm_Z2_module}     symm_Z2_phase: {symm_Z2_phase}")
print(f"trivial_Z2_module: {trivial_Z2_module} trivial_Z2_phase: {trivial_Z2_phase}")

### Training schedule ###

lr_schedule = scheduler_initializer(schedule['name'], schedule)

optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
ds_schedule = optax.linear_schedule(1e-1, 1e-4, epochs)
SR = nk.optimizer.SR(diag_shift=ds_schedule)

E_ED = None
x_ED = None

field_list=[[0.0,0.001,0.001], [-1.0,0.001,0.001]]
coupling_list=[
    [0.0, -10, -4],
    [0.0, -4, -10],
    [0.0, -4, -4],
    [0.0, 4, 4],
    [0.0, 4, 10],
    [0.0, 10, 4]
]

for i, size in enumerate(sizes):

    N = int(np.prod(size)) 
    if N > 20: exact_diag = False

    write_folder = write +  f"Size_{size[0]}x{size[1]}/"

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

    for fields in field_list:

        for couplings in coupling_list:

            for token_size, embedding_d, n_heads, n_blocks, n_ffn_layers in zip(
                [[2,1]] , 
                [32], 
                [2], 
                [2], 
                [2]):

                # for token_size in [[2,1],[2,2]]:
                #     for embedding_d in [32]:
                #         for n_heads in [2,4,8]:
                #             for n_blocks in [1,2]:
                #                 for n_ffn_layers in [2,4]:

                # for j, (field, ZZ_hop, token_size, embedding_d, n_heads, n_blocks, n_ffn_layers) in enumerate(zip(
                #         [ [0.0,0.001,0.001], [0.0,0.001,0.001], [0.0,0.001,0.001], [-1.0,0.001,0.001], [-1.0,0.001,0.001], [-1.0,0.001,0.001]],
                #         [ [0.0,], [-1], [-0.2], [0.4], [1.5], [3.5]],
                #         [[2,1],[2,1], [2,1],[2,1], [2,1],[2,1]] , 
                #         [32, 32, 32, 32, 32, 32], 
                #         [2, 2, 2, 2, 2, 2], 
                #         [2, 2, 2, 2, 2, 2], 
                #         [2, 2, 2, 2, 2, 2])):   
                    
                ## Update Hamiltonian
                field_terms= [ (fields[0],'X'), (fields[1],'Y'), (fields[2],'Z')]
                coupling_terms= [ (couplings[0],'XX','NN'), (couplings[1],'YY','NN2'), (couplings[2],'ZZ','NN')]

                chain = Chain(size[0], **kwargs_lattice)
                chain.plot_lattice(1)
                cm = GeneralNeighborCoupling(chain, field_terms, coupling_terms)
                H = Runner(cm).build_hamiltonian()

                print(f"\n---- Parameters: Size {size} Fields: {fields}  Couplings: {couplings} ----\n\n")
                print(f"Token size: {token_size} \nEmbedding D: {embedding_d} \nHeads: {n_heads}\n")
                print(f"Blocks: {n_blocks} \nffn_layers: {n_ffn_layers}\n\n")

                if exact_diag:# and not os.path.isfile(write_folder + f"Oxalate_xED_{size[0]}x{size[1]}_strength_{strength:.1f}_theta_{theta:.1f}_phi_{phi:.1f}.txt"):
                    print("Running exact diagonalization...")
                    E_ED, x_ED = Runner(cm).exact_energy_lanczos(eigenstates=True)
                    E_ED = float(E_ED.squeeze(-1))
                    print(f"Energy ED: {E_ED}")

                callback_artifacts = {}
                time_in = time.time()

                model = SplitTraining_ViT_MLP(
                    train_phase = False,

                    lattice_size=tuple(size),
                    token_size=tuple(token_size),
                    embedding_d=embedding_d,
                    n_heads=n_heads,
                    n_blocks=n_blocks,
                    n_ffn_layers=n_ffn_layers,
                    final_architecture=final_architecture,
                    is_complex=is_complex,
                    symm_2D_module = symm_2D_module,
                    symm_Z2_module = symm_Z2_module,
                    trivial_Z2_module = trivial_Z2_module,

                    param_dtype_phase = jnp.float64,
                    hidden_alpha = tuple(alphas),
                    activation = tuple(activations),
                    output_dim = output_dim,
                    symm_2D_phase = symm_2D_phase,
                    symm_Z2_phase = symm_Z2_phase,
                    trivial_Z2_phase = trivial_Z2_phase
                )

                vstate = nk.vqs.MCState(
                    sampler,
                    model,
                    n_samples=n_samples,
                    n_discard_per_chain=0,
                    chunk_size=None
                )
                
                gs = nk.driver.VMC(
                    H,
                    optimizer,
                    variational_state=vstate,
                    preconditioner=SR
                )


                log = (
                    nk.logging.RuntimeLog()
                )  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.
                keeper = BestIterKeeper(H, N, 1e-8)

                # keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.
                print(f"\nTraining module...")
                print(f"model.train_phase = {model.train_phase}")
                gs.run(n_iter=epochs, out=log, callback=[keeper.update], show_progress=True)
                mean, std, psi  = phase_stats_vstate(vstate)
                print(f"VS phase: {mean} \u00b1 {std}  ({psi})\n \n")
                print(f"Module Trained. Freezing module and training phase...")

                model = SplitTraining_ViT_MLP(
                    train_phase = True,

                    lattice_size=tuple(size),
                    token_size=tuple(token_size),
                    embedding_d=embedding_d,
                    n_heads=n_heads,
                    n_blocks=n_blocks,
                    n_ffn_layers=n_ffn_layers,
                    final_architecture=final_architecture,
                    is_complex=is_complex,
                    symm_2D_module = symm_2D_module,
                    symm_Z2_module = symm_Z2_module,
                    trivial_Z2_module = trivial_Z2_module,

                    param_dtype_phase = jnp.float64,
                    hidden_alpha = tuple(alphas),
                    activation = tuple(activations),
                    output_dim = output_dim,
                    symm_2D_phase = symm_2D_phase,
                    symm_Z2_phase = symm_Z2_phase,
                    trivial_Z2_phase = trivial_Z2_phase
                )
                rng_module=vstate.sampler_state.rng
                samples_last = vstate.sample(n_samples=n_samples).reshape(n_samples, N)
                params_phase_init = model.init(rng_module, samples_last) # Inicializacion de parametros de fase
                params_module = vstate.parameters  # Parametros del vstate con modulo entrenado
                
        

                params_combined = {
                    "BatchedSpinViT_0": jax.tree_util.tree_map(lambda x: jnp.array(x), params_module["BatchedSpinViT_0"]),
                    "BatchedMultiLayerPerceptron_0": params_phase_init["params"]["BatchedMultiLayerPerceptron_0"]
                }

                mask = {"params": {"BatchedSpinViT_0": False, "BatchedMultiLayerPerceptron_0": True}}
                optimizer_phase = optax.masked(optimizer, mask)

                vstate = nk.vqs.MCState(
                    sampler,
                    model,
                    n_samples=n_samples,
                    n_discard_per_chain=0,
                    chunk_size=None
                )
                vstate.parameters = params_combined
                
                gs = nk.driver.VMC(
                    H,
                    optimizer,
                    variational_state=vstate,
                    preconditioner=SR
                )

                gs.run(n_iter=epochs, out=log, callback=[keeper.update], show_progress=True)

                vstate=keeper.best_state

                time_out = time.time()
                time_exe= time_out - time_in
                
                if exact_diag:
                    keeper.E_ED = E_ED
                    log.E_ED = E_ED

                ## Save results
                flat_fields=''
                field_values=''
                title_label=''
                for f,v in zip(['X','Y','Z'], fields):
                    flat_fields += f + '_'
                    field_values += f"{v}" + '_'
                    title_label += f"{f}:{v}  "

                flat_couplings=''
                coupling_values=''
                for f,v in zip(['XX','YY','ZZ'], couplings):
                    flat_couplings += f + '_'
                    coupling_values += f"{v}" + '_'
                    title_label += f"{f}:{v}  "
                
                coup_label = f"{size[0]}x{size[1]}_fields_{flat_fields}_{field_values}_couplings_{flat_couplings}_{coupling_values}"
                nn_label = f"b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}"

                sim_label = f"Ising_simulation_" + coup_label + nn_label
                ED_label = f"Ising_xED_" + coup_label
                json_label = f"Ising_results_" + nn_label
                title_label_callback = f"Callback  "+ title_label + f"  ({size[0]}x{size[1]})"

                
                if dump_simulation:
                    # For plotting architecture
                    architecture = f"|| b: {token_size}  D_emb: {embedding_d}  heads: {n_heads} ||\n"
                    architecture += f"|| n_blocks: {n_blocks}   ffn_layers: {n_ffn_layers} ||\n"

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
                        'name': 'Chain/Square',
                        'size': size, 
                        'bc': kwargs_lattice['bc'],
                    },
                    'coupling_model': {
                        'operators': cm.operators,
                        'field': fields,
                        'couplings': couplings
                    },

                    'model_NN': {
                        "name": 'SplitTraining_ViT_MLP',

                        "lattice_size":size,
                        "token_size":token_size,
                        "embedding_d":embedding_d,
                        "n_heads":n_heads,
                        "n_blocks":n_blocks,
                        "n_ffn_layers":n_ffn_layers,
                        "final_architecture":final_architecture,
                        "is_complex":is_complex,
                        "symm_2D_module" : symm_2D_module,
                        "symm_Z2_module" : symm_Z2_module,
                        "trivial_Z2_module" : trivial_Z2_module,

                        "param_dtype_phase" : "jnp.float64",
                        "hidden_alpha" : alphas,
                        "activation" : activation_name,
                        "output_dim" : output_dim,
                        "symm_2D_phase" : symm_2D_phase,
                        "symm_Z2_phase" : symm_Z2_phase,
                        "trivial_Z2_phase" : trivial_Z2_phase
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
                    sim_label = sim_label,
                    ED_label=ED_label,
                    json_label=json_label
                )





    

