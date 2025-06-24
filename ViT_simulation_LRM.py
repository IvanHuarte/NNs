#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from netket.operator.spin import sigmax, sigmaz
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
from VA_project.model.cm import LongRangeModel
from VA_project.lattice.lattice import Square
from VA_project.engine.runners import Runner
from NN_module.sim_utils import (
    save_results, dump_callback
)
from NN_module.NN_utils import scheduler_initializer
from transformer_LR_WF.utils import *
from NN_module.models.ViT_2D import BatchedSpinViT

# Cargamos configuracion de archivo json

with open("/home/ihuarte/Escritorio/Ivan/NNs/config_vit.json",'r') as f:
    config = json.load(f)

strength = config['strength']          # Lattice and coupling model
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
symm_2D = config['symm_2D']
symm_Z2 = config['symm_Z2']
trivial_Z2 = config['trivial_Z2']

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

print(f"symm_2D: {symm_2D}")
print(f"symm_Z2: {symm_Z2}")
print(f"trivial_Z2: {trivial_Z2}")

### Training schedule ###
lr_schedule = scheduler_initializer(schedule['name'], schedule)

optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
ds_schedule = optax.linear_schedule(1e-2, 1e-4, epochs)
SR = nk.optimizer.SR(diag_shift=ds_schedule)

E_ED = None
x_ED = None

for i, size in enumerate(sizes):

    N = int(np.prod(size)) 
    if N > 20: exact_diag = False

    write_folder = write +  f"Oxalate_size_{size[0]}x{size[1]}/"

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

    #for j, (theta, phi) in enumerate(zip(theta_list, phi_list)):
    for j, (XZ_field, ZZ_hop, token_size, embedding_d, n_heads, n_blocks, n_ffn_layers) in enumerate(zip(
            [ [-1.0,0.001], [-1.0,0.001], [-1.0,0.001], [-1.0,0.001], [-1.0,0.001], [-1.0,0.001]],
            [ [-20], [-1], [-0.2], [0.4], [1.5], [3.5]],
            [[2,1],[2,1], [2,1],[2,1], [2,1],[2,1]] , 
            [32, 32, 32, 32, 32, 32], 
            [2, 2, 2, 2, 2, 2], 
            [2, 2, 2, 2, 2, 2], 
            [2, 2, 2, 2, 2, 2])):   
        
        alpha=2.5
        
        print(f"\n---- Parameters: Size {size}  X={XZ_field[0]:1f}  Z={XZ_field[1]:1f}  ZZ={ZZ_hop[0]:1f} alpha: {alpha}----\n\n")
        print(f"Token size: {token_size} \nEmbedding D: {embedding_d} \nHeads: {n_heads}\n")
        
        ## Update Hamiltonian
        lat = Square(*size, bc='periodic', order = 'default_2')
        lrm = LongRangeModel(lat,fields = XZ_field, hoppings=ZZ_hop, alpha = 2.5)
        H = Runner(lrm).build_hamiltonian()

        if exact_diag:# and not os.path.isfile(write_folder + f"Oxalate_xED_{size[0]}x{size[1]}_strength_{strength:.1f}_theta_{theta:.1f}_phi_{phi:.1f}.txt"):
            print("Running exact diagonalization...")
            E_ED, x_ED = Runner(lrm).exact_energy_lanczos(eigenstates=True)
            E_ED = float(E_ED.squeeze(-1))
            print(f"Energy ED: {E_ED}")

        # for token_size, embedding_d, n_heads, n_blocks, n_ffn_layers in zip(
        #     [[2,1],[2,1], [2,1],[2,1], [2,2], [2,2], [2,2]] , 
        #     [32,32,64,64,32,32,64,64], 
        #     [4,4,4,4,4,4,4,4], 
        #     [1,2,1,2, 1,2,1,2], 
        #     [2,4,2,2, 2,4,2,2]):

        # for token_size in [[2,1],[2,2]]:
        #     for embedding_d in [32]:
        #         for n_heads in [2,4,8]:
        #             for n_blocks in [1,2]:
        #                 for n_ffn_layers in [2,4]:


        callback_artifacts = {}
        time_in = time.time()

        model = BatchedSpinViT(
            lattice_size=tuple(size),
            token_size=tuple(token_size),
            embedding_d=embedding_d,
            n_heads=n_heads,
            n_blocks=n_blocks,
            n_ffn_layers=n_ffn_layers,
            final_architecture=final_architecture,
            is_complex=is_complex,
            symm_2D = symm_2D,
            symm_Z2 = symm_Z2,
            trivial_Z2 = trivial_Z2
        )

        vstate = nk.vqs.MCState(
            sampler,
            model,
            n_samples=n_samples,
            n_discard_per_chain=0,
            chunk_size=None,
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
        gs.run(n_iter=epochs, out=log, callback=[keeper.update], show_progress=True)

        vstate=keeper.best_state

        time_out = time.time()
        time_exe= time_out - time_in
        
        if exact_diag:
            keeper.E_ED = E_ED
            log.E_ED = E_ED

        ## Save results

        coup_label = f"{size[0]}x{size[1]}_XZ_{XZ_field[0]}_{XZ_field[1]}_J_{ZZ_hop[0]}_alpha_{alpha}_"
        nn_label = f"b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}"

        sim_label = f"LRM_simulation_" + coup_label + nn_label
        ED_label = f"LRM_xED_" + coup_label
        json_label = f"LRM_results_" + coup_label + nn_label
        title_label_callback = f"Callback  X:{XZ_field[0]} Z:{XZ_field[1]} J:{ZZ_hop[0]} " + r"$\alpha=%.2f$"%alpha + f" ({size[0]}x{size[1]})"

        if dump_simulation:
            # For plotting architecture
            
            #architecture = f"|| X: {XZ_field[0]} Z:{XZ_field[1]} J:{ZZ_hop[0]} alpha: {alpha} ||\n"
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

        if exact_diag:
            error=float(np.abs(E_best-E_ED)/np.abs(E_ED))
        else:
            E_ED = None
            x_ED = None
            error = None    

        # Save the results

        dump_setup ={

            'lattice':{
                'name': 'Chain/Square',
                'size': size, 
                'bc': kwargs_lattice['bc'],
            },
            'coupling_model': {
                'X':XZ_field[0],
                'Y':XZ_field[1],
                'J':ZZ_hop[0],
                'alpha': alpha
            },

            'model_NN': {
                'name': 'ViT',
                "lattice_size": size,
                "token_size": token_size,
                "embedding_d": embedding_d,
                "n_heads": n_heads,
                "n_blocks": n_blocks,
                "n_ffn_layers": n_ffn_layers,
                "final_architecture": final_architecture,
                'is_complex': is_complex,
                'symm_2D' : symm_2D,
                'symm_Z2' : symm_Z2,
                'trivial_Z2' : trivial_Z2
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
