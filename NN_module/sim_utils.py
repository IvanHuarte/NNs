import os
import copy
import json
import ast
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import flax
import jax
import jax.numpy as jnp
import netket as nk
import numpy.typing as npt
from flax.serialization import to_bytes, from_bytes
from typing import Optional
from datetime import date
from platform import architecture, python_version
import pathlib
from NN_module.models.MLP import BatchedMultiLayerPerceptron
from NN_module.models.ViT_2D import BatchedSpinViT
from NN_module.models.split_training import SplitTraining_ViT_MLP, SplitTraining_ViT_CNN
from NN_module.NN_utils import (
    activation_dict, sampler_dict, rule_dict
)
class BestIterKeeper:
    """Store the values of a bunch of quantities from the best iteration.

    "Best" is defined in the sense of lowest energy.

    Args:
        Hamiltonian: An array containing the Hamiltonian matrix.
        N: The number of spins in the chain.
        baseline: A lower bound for the V score. If the V score of the best
            iteration falls under this threshold, the process will be stopped
            early.
        filename: Either None or a file to write the best state to.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        varstep = self.N * var / mean**2
        #print(f" Variance: {var}, Mean: {mean}, Vscore: {varstep}") 

        if self.best_energy > energystep:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(driver.state.parameters)
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.vscore > self.baseline
    
def get_write_folder_from_model(config):

    name = config['model_NN']['selection']
    model_setup = config['model_NN'][name]
    model_label= config['CM']['selection'] +'_'+ name

    if  name == "SplitTraining_ViT_MLP": 
        
        conditions = [model_setup['symm_2D_module'], model_setup['symm_2D_phase'], 
                      model_setup['symm_Z2_module'], model_setup['symm_Z2_phase']]
        labels = ["_2DM","_2DP","_Z2M","_Z2P"]
  

    elif  name == "SplitTraining_ViT_CNN":
        conditions = [model_setup['symm_2D_module'], model_setup['symm_Z2_module']]
        labels = ["_2DM","_Z2M"]

    symm=''
    for cond, label in zip(conditions,labels):
        if cond:
            symm+=label
            if label == '_Z2M':  symm += "t" if model_setup['trivial_Z2_module'] else "nt"
            if label == '_Z2P':  symm += "t" if model_setup['trivial_Z2_phase'] else "nt"
        
    return config['write_folder_sim'] + model_label+ symm + "/"


from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

def display_simulation_settings(settings, n_cols=5):
    console=Console()
    cm_sel = settings["CM"]["selection"]
    nn_sel = settings["model_NN"]["selection"]

    cm_color= "red"
    nn_color= "bright_yellow"
    
    # Título principal
    title_text = f"[bold blue]🔧 CM:[/] [{cm_color}]{cm_sel}[/]   [bold blue]🧠 NN_architecture:[/] [{nn_color}]{nn_sel}[/]"
    console.rule(title_text)

    def _group_params(params_dict):
        # Filtra claves no relevantes
        items = [(k, v) for k, v in params_dict.items() if not k.endswith("_list")]
        grouped = [items[i:i + n_cols] for i in range(0, len(items), n_cols)]

        table = Table(show_header=False, box=None, pad_edge=False)
        for i in range(n_cols):
            table.add_column(justify="left")
        for group in grouped:
            row = [f"[bold]{k}[/]= {v}" for k, v in group]
            while len(row) < n_cols:
                row.append("")
            table.add_row(*row)
        return table
    
    # Size
    size = settings.get("size", None)
    if isinstance(size, (list, tuple)) and all(isinstance(x, int) for x in size):
        size_str = "x".join(map(str, size))
        console.print(f"[bold yellow]🧱 Size:[/] [cyan]{size_str}[/]\n")

    # Panel de acoplamiento
    cm_dict = settings["CM"].get(cm_sel, {})
    if isinstance(cm_dict, dict):
        table_cm = _group_params(cm_dict)
        panel_cm = Panel(table_cm, title=f"[bold]{cm_sel}[/]", border_style=cm_color)
        console.print(panel_cm)

    # Panel de red neuronal
    nn_dict = settings["model_NN"].get(nn_sel, {})
    if isinstance(nn_dict, dict):
        table_nn = _group_params(nn_dict)
        panel_nn = Panel(table_nn, title=f"[bold]{nn_sel}[/]", border_style=nn_color)
        console.print(panel_nn)

    console.rule("[bold green]")



def architecture_label(name, model_setup):

    if name == 'MLP':
        set_dim=[f"D({dim})" for dim in model_setup['hidden_alpha']] 
        set_act=[f"A({act})" for act in model_setup['activation']]
        setup='||'
        for i in range(len(model_setup['hidden_alpha'])):
            setup+=f" {set_dim[i]} |"
            setup+=f" {set_act[i]} |" if '0' not in set_act[i] else ''
        setup+="|"
        architecture = setup
        return architecture

    elif name == 'ViT':
        architecture = f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        return architecture
    
    elif name == 'SplitTraining_ViT_MLP':
        architecture = f"ViT \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"MLP \n"
        set_dim=[f"D({dim})" for dim in model_setup['hidden_alpha']] 
        set_act=[f"A({act})" for act in model_setup['activation']]
        setup='||'
        for i in range(len(model_setup['hidden_alpha'])):
            setup+=f" {set_dim[i]} |"
            setup+=f" {set_act[i]} |" if '0' not in set_act[i] else ''
        setup+="|"
        architecture += setup
        return architecture
    
    elif name == 'SplitTraining_ViT_CNN':
        architecture = f"ViT \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"CNN\n"
        architecture += f"|| block_channels: {model_setup['block_channels']}    kernel:{model_setup['kernel_size']}    n_ffn_lay:{model_setup['n_ffn_layers_cnn']} ||"
        return architecture
    
def get_filenames_from_settings(cm_name, nn_name, **kwargs):
    
    model_label = cm_name + '_' + nn_name
    size=kwargs['size']

    if cm_name == 'Oxalate':
        strength = kwargs['strength'] ; theta = kwargs['theta'] ; phi = kwargs['phi']
        cparams= f"_strength_{strength}_theta_{theta}_phi_{phi}"
        call_params = r"$a = %.1f$  $\theta = %.1f$  $\phi = %.1f$"%(strength,theta,phi)

    elif cm_name == 'Chain_YYZZ':
        fields = kwargs['fields'] ; couplings = kwargs['couplings']
        flat_fields=''
        field_values=''
        call_params=''
        for f,v in zip(['X','Y','Z'], fields):
            flat_fields += f + '_'
            field_values += f"{v}" + '_'
            call_params += f"{f}:{v}  "

        flat_couplings=''
        coupling_values=''
        for f,v in zip(['XX','YY','ZZ'], couplings):
            flat_couplings += f + '_'
            coupling_values += f"{v}" + '_'
            call_params += f"{f}:{v}  "

    if nn_name == 'MLP':
        alphas = kwargs['hidden_alpha'] ; activation = kwargs['activation'] 
        act_label=''
        dim_label=''
        for act in activation:
            act_label += f"{act}_"
        for a in alphas:
            dim_label += f"{a}_"
        nnparams = f"_alphas_{dim_label}_activations_{act_label}"

    elif nn_name == 'ViT':
        token_size = kwargs['token_size'] ; embedding_d = kwargs['embedding_d'] ; n_heads = kwargs['n_heads'] 
        n_blocks = kwargs['n_blocks'] ; n_ffn_layers = kwargs['n_ffn_layers']
        nnparams = f"_b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}"
    
    elif nn_name == 'SplitTraining_ViT_MLP':
        token_size = kwargs['token_size'] ; embedding_d = kwargs['embedding_d'] ; n_heads = kwargs['n_heads']   # ViT
        n_blocks = kwargs['n_blocks'] ; n_ffn_layers = kwargs['n_ffn_layers']
        alphas = kwargs['hidden_alpha'] ; activation = kwargs['activation']     # MLP
        act_label=''
        dim_label=''
        for act in activation:
            act_label += f"{act}_"
        for a in alphas:
            dim_label += f"{a}_"
        
        nnparams= f"_ViT_b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}__MLP_alphas_{dim_label}_activations_{act_label}"
        
    
    elif nn_name == 'SplitTraining_ViT_CNN':
        token_size = kwargs['token_size'] ; embedding_d = kwargs['embedding_d'] ; n_heads = kwargs['n_heads']   # ViT
        n_blocks = kwargs['n_blocks'] ; n_ffn_layers = kwargs['n_ffn_layers']
        block_channels = kwargs['block_channels'] ; kernel_size = kwargs['kernel_size'] ; n_ffn_layers_cnn = kwargs['n_ffn_layers_cnn']     # CNN
        cha_label=''
        for ch in block_channels:
            cha_label += f"{ch}_"
        nnparams = f"_channels_{cha_label}kernel_{kernel_size[0]}x{kernel_size[1]}_n_ffn_lay_{n_ffn_layers_cnn}"

    

    sim_label = model_label + f"_simulation_{size[0]}x{size[1]}"+ cparams + nnparams
    ED_label = model_label + f"_xED_{size[0]}x{size[1]}"+cparams
    json_label = model_label+f"_results_{size[0]}x{size[1]}"+ cparams + nnparams
    title_label_callback = f"Callback "+model_label+" " + call_params + f"  ({size[0]}x{size[1]})"

    return sim_label, ED_label, json_label, title_label_callback

        
        
def dump_callback(logger, settings, write = False):

    callback_artifacts = {}

    time_exe = settings['time_exe']
    write_folder = settings['write_folder']
    architecture_display = settings['architecture']
    opt_name = settings['opt_name']
    learning_rate = settings['learning_rate']
    sim_label = settings['sim_label']
    title_label_callback= settings['title_label_callback']

    N=int(np.prod(settings['size']))

    os.makedirs(write_folder, exist_ok = True)
    
    # Extract some results
    E_hist = np.array(logger['Energy']['Mean']).real
    dev_E_hist = np.array(logger['Energy']['Sigma']).real
    E_best = min(logger['Energy']['Mean']).real


    if hasattr(logger, 'E_ED'):
        E_gr = np.array(logger.E_ED).real
        error=np.abs(E_hist-E_gr)/np.abs(E_gr)

    # Calculate the variance score
    var= np.real(np.array(logger['Energy']['Variance']))
    vscore = N*var/(E_hist**2)
    #vs_min = np.round(np.log10(np.min(vscore)))-1

    setup_sim = f"E_best: {E_best:.4f} \nopt: {opt_name} \nl_rate: {learning_rate} \ntime_exe: {time_exe:.2f}"
    if hasattr(logger, 'E_ED'):
        setup_sim = f"E_ED: {E_gr:.4f}\n" + setup_sim 
    
    # Plot
    if hasattr(logger, 'E_ED'):
        _, ax = plt.subplots(3,1,figsize=(8, 18))
        v = 2 ; e = 1
    else:
        _, ax = plt.subplots(2,1,figsize=(8, 12))
        v = 1 ; e = 2

    ax[0].set_title(title_label_callback)

    if hasattr(logger, 'E_ED'):
        ax[0].errorbar(range(len(E_hist)), E_hist, yerr=dev_E_hist, fmt='none', ecolor='r', label='E_stdev')
        ax[0].hlines(E_gr,0,len(E_hist), color='green', label='ED Energy')
                    
    ax[0].plot(E_hist, color='blue', label='E')
    ax[0].text(0.45, 0.85, architecture_display, transform=ax[0].transAxes, fontsize=12, color='k', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax[0].text(0.9, 0.75, setup_sim, transform=ax[0].transAxes, fontsize=10, color='k', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax[0].legend()
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Energy', fontsize=12)
    ax[0].grid()


    ax[v].plot(vscore, color='purple', label='Vscore')
    ax[v].set_yscale('log')
    #ax[v].set_ylim(bottom=vs_min)
    ax[v].legend()
    ax[v].set_xlabel('Iteration')
    ax[v].set_ylabel('Vscore', fontsize=12)
    ax[v].grid()

    if hasattr(logger,'E_ED'):
        ax[e].plot(error, color='red', label='E')
        ax[e].set_yscale('log')
        ax[e].legend()
        ax[e].set_xlabel('Iteration')
        ax[e].set_ylabel('Error', fontsize=12)
        ax[e].grid()

    file_path = write_folder + f"Callback_" + sim_label
    figure_path = file_path + ".jpeg"

    plt.tight_layout()
    plt.savefig(figure_path, dpi=600)
    plt.close()

    callback_artifacts['plot'] = figure_path

    # Save the data
    if write:
        E_path = file_path + "_E_hist.txt"
        error_path = file_path + "_error.txt"
        vscore_path = file_path + "_vscore.txt"

        np.savetxt(E_path, np.array(E_hist))
        np.savetxt(error_path, np.array(error))
        np.savetxt(vscore_path, np.array(vscore))

        callback_artifacts['Energy'] = E_path
        callback_artifacts['error'] = error_path
        callback_artifacts['vscore'] = vscore_path
    
    return callback_artifacts

def save_results(vstate, setup, x_ED = None, modphase=None, modphase_ED=None, write_folder = './', sim_label = '', ED_label = '', json_label = ''):
    """Save the results of the simulation.
    Args:
        vstate: The variational state.
        setup: The setup dictionary containing the simulation parameters.
        x_ED: The exact diagonalization results (optional).
        write_folder: The folder to save the results.
        sim_label: A label for the simulation.
    """

    os.makedirs(write_folder, exist_ok = True)

    size = setup['lattice']['size']
    
    # Save the variational state
    file = f"{sim_label}"
    file_ED =  f"{ED_label}"

    path_vstate = write_folder + file + "_vstate_params.msgpack"

    with open(path_vstate, "wb") as f:
        f.write(to_bytes(vstate.parameters))
    
    setup['_artifacts']['vstate'] = path_vstate

    # Save exact diagonalization eigenstate 
    if x_ED is not None:
        path_ED = write_folder + file_ED + ".txt"
        if not os.path.isfile(write_folder + file_ED):
            np.savetxt(path_ED, x_ED)
        setup['_artifacts']['x_ED'] = path_ED

    # Save modulus and phase from vstate and/or xED
    setup['_artifacts']['modphase']={}
    if modphase is not None:
        modphase_path = write_folder + file + "_modphase_vstate.txt"
        np.savetxt(modphase_path, modphase)
        setup['_artifacts']['modphase']['vstate'] = modphase_path

    if modphase_ED is not None:
        modphase_ED_path = write_folder + file_ED + "_modphase_xED.txt"
        np.savetxt(modphase_ED_path, modphase_ED)
        setup['_artifacts']['modphase']['xED'] = modphase_ED_path

    #Write metadata in main setup artifact
    metadata = {
        "date": date.today().strftime("%x"),
        "architecture": architecture(),
        "versions": {
            "python": python_version(),
            "numpy": np.__version__,
            "scipy": sp.__version__,
            "netket": nk.__version__,
            "jax": jax.__version__,
            "flax": flax.__version__
        },
        "device": str(jax.devices()[0])
    }

    setup['metadata'] = metadata
    setup['results']['vstate'] = path_vstate

    # Save the main artifact
    setup_path = write_folder + json_label + ".json"
    with open(setup_path, "w") as outfile:
        json.dump(setup, outfile, separators=(",", ":"), sort_keys=True, indent=4)



def init_model(name, model_setup):

    if name == 'MLP':
        activation=model_setup['activation']
        if all([type(act) in [str, int] for act in activation]):
            activation = tuple([activation_dict[act] if act != 0 else 0 for act in activation])

        return BatchedMultiLayerPerceptron(
            lattice_size=tuple(model_setup['lattice_size']),
            hidden_alpha=tuple(model_setup['hidden_alpha']),
            activation=activation,
            param_dtype=jnp.complex128,
            output_dim=1,
            symm_2D=model_setup['symm_2D'],
            symm_Z2=model_setup['symm_Z2'],
            trivial_Z2=model_setup['trivial_Z2']
        )
    
    if name == 'ViT':
        return BatchedSpinViT(
            lattice_size=tuple(model_setup['lattice_size']),
            token_size=tuple(model_setup['token_size']),
            embedding_d=model_setup['embedding_d'],
            n_heads=model_setup['n_heads'],
            n_blocks=model_setup['n_blocks'],
            n_ffn_layers=model_setup['n_ffn_layers'],
            final_architecture=tuple(model_setup['final_architecture']),
            is_complex=model_setup['is_complex'],
            symm_2D = model_setup['symm_2D'],
            symm_Z2 = model_setup['symm_Z2'],
            trivial_Z2 = model_setup['trivial_Z2']
        )
    if name == 'SplitTraining_ViT_MLP':

        activation=model_setup['activation']
        if all([type(act) in [str, int] for act in activation]):
            activation = tuple([activation_dict[act] if act != 0 else 0 for act in activation])
        return SplitTraining_ViT_MLP(

                lattice_size=tuple(model_setup['lattice_size']),
                token_size=tuple(model_setup['token_size']),
                embedding_d=model_setup['embedding_d'],
                n_heads=model_setup['n_heads'],
                n_blocks=model_setup['n_blocks'],
                n_ffn_layers=model_setup['n_ffn_layers'],
                final_architecture=ast.literal_eval(model_setup['final_architecture']),
                is_complex=model_setup['is_complex'],
                symm_2D_module = model_setup['symm_2D_module'],
                symm_Z2_module = model_setup['symm_Z2_module'],
                trivial_Z2_module = model_setup['trivial_Z2_module'],

                param_dtype_phase = jnp.float64,
                hidden_alpha = tuple(model_setup["hidden_alpha"]),
                activation = tuple(activation),
                output_dim = model_setup["output_dim"],
                symm_2D_phase = model_setup["symm_2D_phase"],
                symm_Z2_phase = model_setup["symm_Z2_phase"],
                trivial_Z2_phase = model_setup["trivial_Z2_phase"]
        )
    if name == 'SplitTraining_ViT_CNN':
        return SplitTraining_ViT_CNN(

            lattice_size=tuple(model_setup['lattice_size']),
            token_size=tuple(model_setup['token_size']),
            embedding_d=model_setup['embedding_d'],
            n_heads=model_setup['n_heads'],
            n_blocks=model_setup['n_blocks'],
            n_ffn_layers=model_setup['n_ffn_layers'],
            final_architecture=ast.literal_eval(model_setup['final_architecture']),
            is_complex=model_setup['is_complex'],
            symm_2D_module = model_setup['symm_2D_module'],
            symm_Z2_module = model_setup['symm_Z2_module'],
            trivial_Z2_module = model_setup['trivial_Z2_module'],

            block_channels=tuple(model_setup['block_channels']),
            kernel_size=tuple(model_setup['kernel_size']),
            n_ffn_layers_cnn=model_setup['n_ffn_layers_cnn']
        )
    
def print_tree_keys(obj, indent=0):
    prefix = '  ' * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{prefix}{key}")
            print_tree_keys(value, indent + 1)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            print(f"{prefix}- [{i}]")
            print_tree_keys(item, indent + 1)

def load_vstate(setup, tree_data=False):

    # Initialize model
    N = int(np.prod( setup['lattice']['size'] ))
    model = init_model(setup["model_NN"]["name"],setup["model_NN"]["setup"])

    # Initialize hilbert space
    hi = nk.hilbert.Spin(s=0.5, N = N)

    #Dummy init to have the vs_params structure
    rng = jax.random.PRNGKey(0)
    dummy_params = model.init(rng, jnp.ones((1, N), dtype=jnp.complex64))
    
    # Initialize sampler
    sampler_name = setup['sampler']['name']
    if 'rules' in setup['sampler']:
        rule = rule_dict[setup['sampler']['rules']]
        sampler = sampler_dict[sampler_name](hi, rule=rule, dtype=jnp.complex64)

    else:
        sampler = sampler_dict[sampler_name](hi, dtype = jnp.complex64)

    # Load parameters and initialize vstate
    restored_rng = jnp.array(setup['sampler']['rng'], dtype=jnp.uint32)
    n_samples = setup['sampler']['n_samples']

    vs = nk.vqs.MCState(sampler, model, n_samples = n_samples, seed = 0)
    dummy_params=dummy_params["params"]    

    if tree_data:      # For debugging
        import msgpack
        with open(setup['_artifacts']['vstate'], "rb") as f:
            data = msgpack.unpack(f, raw=False)
        print_tree_keys(data)


    vs_path = setup['results']['vstate']
    with open(vs_path, "rb") as f:
        loaded_params = from_bytes(dummy_params, f.read())
        vs.parameters = loaded_params

    vs.sampler_state = sampler.init_state(model, {"params":vs.parameters}, restored_rng)

    return vs
