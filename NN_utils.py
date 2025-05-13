import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import netket as nk
import json
import flax
import flax.linen as nn
import jax
from flax.serialization import to_bytes
import copy
import numpy.typing as npt
from typing import Optional
import pathlib
from datetime import date
from platform import architecture, python_version
import os

activation_dict={
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'softmax': nn.softmax,
    'gelu': nn.gelu,
    'swish': nn.swish,
    'selu': nn.selu,
    'elu': nn.elu,
    'softplus': nn.softplus,
    'relu': nn.relu
}

optimizer_name_dict={
    'Sgd': nk.optimizer.Sgd,
    'adam': nk.optimizer.Adam,
    'AdaGrad': nk.optimizer.AdaGrad
}

sampler_dict={
    'MetropolisLocal': nk.sampler.MetropolisLocal,
    'MetropolisExchange': nk.sampler.MetropolisExchange,
    'MetropolisHamiltonian': nk.sampler.MetropolisHamiltonian
}


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

        if self.best_energy > energystep:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(driver.state.parameters)
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.vscore > self.baseline


def dump_callback(logger, settings, plot =  True, write = False):

    size = settings['size']
    theta = settings['theta']
    phi = settings['phi']
    time_exe = settings['time_exe']
    write_folder = settings['write_folder']
    architecture_display = settings['architecture']
    opt_name = settings['optimizer']
    learning_rate = settings['learning_rate']
    sim_label = settings['sim_label']
    
    # Extract some results
    E_hist = np.array(logger['Energy']['Mean']).real
    dev_E_hist = np.array(logger['Energy']['Sigma']).real
    E_best = min(logger['Energy']['Mean']).real
    if hasattr(logger, 'E_ED'):
        E_gr = np.array(logger['E_ED']).real
        error=np.abs(E_hist-E_gr)/np.abs(E_gr)

    # Calculate the variance score
    var= np.array(logger['Energy']['Variance']).real
    vscore = int(np.prod(settings['size']))*var//(E_hist**2)

    setup_sim = f"E_best: {E_best:.4f} \nopt: {opt_name} \nl_rate: {learning_rate} \ntime_exe: {time_exe:.2f}"
    if hasattr(logger, 'E_ED'):
        setup_sim = setup_sim + f" \nE_ED: {E_gr:.4e}"

    if plot:

        _, ax = plt.subplots(3,1,figsize=(8, 18))
        ax[0].set_title(f"Convergence 4x4 theta=9 phi=72")
        ax[0].errorbar(range(len(E_hist)), E_hist, yerr=dev_E_hist, fmt='none', ecolor='r', label='E_stdev')
        ax[0].plot(E_hist, color='blue', label='E')

        if hasattr(logger, 'E_ED'):
            ax[0].hlines(E_gr,0,len(E_hist), color='green', label='ED Energy')

        ax[0].text(0.45, 0.93, architecture_display, transform=ax[0].transAxes, fontsize=12, color='k', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax[0].text(0.9, 0.75, setup_sim, transform=ax[0].transAxes, fontsize=10, color='k', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax[0].legend()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Energy', fontsize=12)
        ax[0].grid()

        ax[1].plot(error, color='red', label='E')
        ax[1].set_yscale('log')
        ax[1].legend()
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Error', fontsize=12)
        ax[1].grid()


        ax[2].plot(vscore, color='purple', label='Vscore')
        ax[2].set_yscale('log')
        ax[2].legend()
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('Vscore', fontsize=12)
        ax[2].grid()
        plt.tight_layout()
        plt.savefig(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}.jpeg", dpi=600)
        plt.close()

    # Save the data
    if write:
        np.savetxt(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_E_hist.txt",
                np.array(E_hist))
        np.savetxt(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_error.txt",
                np.array(error))
        np.savetxt(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_vscore.txt",
                np.array(vscore))
        

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
    path_vstate = write_folder + "Oxalate_vstate_params_" + file + ".npz"

    with open(path_vstate, "wb") as f:
        f.write(to_bytes(vstate.parameters))


    #Save simulation results, data and metadata
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

    setup_path = write_folder + "Oxalate_results_" + file + ".json"
    with open(setup_path, "w") as outfile:
        json.dump(setup, outfile, separators=(",", ":"), sort_keys=True, indent=4)
    
    # Save exact diagonalization eigenstate 
    if x_ED is not None: 
        path_ED = write_folder + "Oxalate_ED_" + file + ".txt"
        np.savetxt(path_ED, x_ED)

def load_vstate(setup):

    sampler_name = setup['']['']
    model = setup

    vs = nk.vqs.MCState(sampler, model, n_samples=..., seed=...)

    vs_path = setup['results']['vstate']

    with open(vs_path, "rb") as f:
        vs.parameters = from_bytes(vs.parameters, f.read())




        


