import os
import copy
import json
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
from NN_module.models.MLP import MultiLayerPerceptron
from NN_module.models.ViT_2D import BatchedSpinViT
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


def dump_callback(logger, settings, write = False):

    callback_artifacts = {}

    size = settings['size']
    theta = settings['theta']
    phi = settings['phi']
    time_exe = settings['time_exe']
    write_folder = settings['write_folder']
    architecture_display = settings['architecture']
    opt_name = settings['opt_name']
    learning_rate = settings['learning_rate']
    sim_label = settings['sim_label']
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

    ax[0].set_title(f"Callback  "+ r"$\theta = %.1f$  $\phi = %.1f$"%(theta,phi) + f"({size[0]}x{size[1]})")

    if hasattr(logger, 'E_ED'):
        ax[0].errorbar(range(len(E_hist)), E_hist, yerr=dev_E_hist, fmt='none', ecolor='r', label='E_stdev')
        ax[0].hlines(E_gr,0,len(E_hist), color='green', label='ED Energy')
                    
    ax[0].plot(E_hist, color='blue', label='E')
    ax[0].text(0.45, 0.93, architecture_display, transform=ax[0].transAxes, fontsize=12, color='k', ha='center', va='center',
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

    file_path = write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}"
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
        

def save_results(vstate, setup, x_ED = None, write_folder = './', sim_label = ''):
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
    strength = setup['coupling_model']['strength']
    theta = setup['coupling_model']['theta']
    phi = setup['coupling_model']['phi']
    
    # Save the variational state
    file = f"{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}_{sim_label}"
    file_ED =  f"{size[0]}x{size[1]}_strength_{strength}_theta_{theta}_phi_{phi}"

    path_vstate = write_folder + "Oxalate_vstate_params_" + file + ".msgpack"

    with open(path_vstate, "wb") as f:
        f.write(to_bytes(vstate.parameters))
    
    setup['_artifacts']['vstate'] = path_vstate

    # Save exact diagonalization eigenstate 
    if x_ED is not None:
        path_ED = write_folder + "Oxalate_xED_" + file_ED + ".txt"
        if not os.path.isfile(write_folder + file_ED):
            np.savetxt(path_ED, x_ED)
        setup['_artifacts']['x_ED'] = path_ED

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
    setup_path = write_folder + "Oxalate_results_" + file + ".json"
    with open(setup_path, "w") as outfile:
        json.dump(setup, outfile, separators=(",", ":"), sort_keys=True, indent=4)



def _init_model(N,model):

    if model['name'] == 'MLP':
        dimensions=model['dense_dim']
        hidden_alpha = tuple([dim//N for dim in dimensions])
        activation=model['activation']
        if all([type(act) in [str, int] for act in activation]):
            activation = tuple([activation_dict[act] if act != 0 else 0 for act in activation])

        return MultiLayerPerceptron(
            N=N,
            hidden_alpha=hidden_alpha,
            activation=activation,
            param_dtype=jnp.complex64,
            output_dim=1
        )
    
    if model['name'] == 'ViT':
        return BatchedSpinViT(
            lattice_size=tuple([6,6]),
            token_size=tuple(model['token_size']),
            embedding_d=model['embedding_d'],
            n_heads=model['n_heads'],
            n_blocks=model['n_blocks'],
            n_ffn_layers=model['n_ffn_layers'],
            final_architecture=tuple(model['final_architecture']),
            is_complex=model['is_complex'],
            symm_2D = True ,#model['symm_2D'],
            symm_Z2 = False ,#model['symm_Z2'],
            trivial_Z2 = True #model['trivial_Z2']
        )

def load_vstate(setup):

    # Initialize model
    N = int(np.prod( setup['lattice']['size'] ))
    model = _init_model(N,setup["model_NN"])

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

    vs_path = setup['results']['vstate']
    with open(vs_path, "rb") as f:
        loaded_params = from_bytes(dummy_params, f.read())
        vs.parameters = loaded_params

    vs.sampler_state = sampler.init_state(model, {"params":vs.parameters}, restored_rng)

    return vs
