import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import netket as nk
from typing import Tuple, Callable, Any, Dict, Optional
import numpy.typing as npt
import copy
import pathlib
import matplotlib.pyplot as plt
import time

from VA_project.model.model import OxalateJKGamma
from VA_project.engine.runners import Runner
from NN_utils import dump_callback, save_results

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
jax.devices()


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
    

class MLP(nn.Module):
    """A simple multi-layer perceptron."""
    
    param_dtype : Any = jnp.complex64
    hidden_dims: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None

    def setup(self):
        assert self.hidden_dims is not None, "Hidden_dims must be provided."

        if self.activation is None:
            self.activation = tuple([0] * len(self.hidden_dims))
        else:
            if len(self.hidden_dims) != len(self.activation):
                raise ValueError("The number of hidden dimensions must match the number of activation functions,"
                                 f" but got {len(self.hidden_dims)} and {len(self.activation)} respectively.")
        
    @nn.compact
    def __call__(self, x):
        
        for hi, act in zip(self.hidden_dims, self.activation):
            x = nn.Dense(hi, param_dtype=self.param_dtype)(x)
            if act:
                x = act(x)

        return x.squeeze(-1)
    

settings={
    
    'size': (4,4), #Lattice size
    'alpha':2.5,

    'NN_arch': {

        'MLP': {

            'rng' : jax.random.PRNGKey(0),
            'layers' : ['dense', 'activation', 'dense'],
            'dimensions' : [128, 128],
            'activation' : [nn.sigmoid],
            'dropout' : [],

        }
    },
    'optimizer': {
        'adam': {
            'learning_rate': 0.01
        },
        'Sgd': {
            'learning_rate': 0.01
        },
        'AdaGrad': {
            'learning_rate': 0.01
        },
    },   

    'sampler': {
        'type': 'MetropolisLocal',
        'n_samples': 1008,
        
    },
}
    
strength = 0.2
theta = 9.0
phi = 72.

diag_shift= 0.001
n_iter=1000


act_dict={
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'softmax': nn.softmax,
    'gelu': nn.gelu,
    'swish': nn.swish,
    'selu': nn.selu,
    'elu': nn.elu,
    'softplus': nn.softplus}

opt_name_dict={
    'Sgd': nk.optimizer.Sgd,
    'adam': nk.optimizer.Adam,
    'AdaGrad': nk.optimizer.AdaGrad} 

opt_name_list = ['Sgd', 'adam', 'AdaGrad']
learning_rate_list = [0.1, 0.01, 0.001]

dimensions_list = [
    (32,32,1),
    (64,64,1),
    (16,32,64,64,32,16,1),
    (64,64,32,16,8,4,2,1)
]
activation_list = [
    [('relu',0,0),(0,'relu',0),('softplus',0,0),(0,'softplus',0)],
    [('relu',0,0),(0,'relu',0),('softplus',0,0),(0,'softplus',0)],
    [(0,'softmax',0,0,0,0,0),(0,'softmax','softmax',0,0,0,0),(0,'softplus',0,0,0,0,0), (0,'softplus','softplus',0,0,0,0)]
]


d=0 ; a=0 

for d, dimensions in enumerate(dimensions_list): 
    
    for a, activation_name in enumerate(activation_list[d]):

        activation = tuple([act_dict[act] if act != 0 else 0 for act in activation_name])

        for opt_name in opt_name_list:

            for learning_rate in learning_rate_list:

                print(f"dimensions: {dimensions} \nactivation: {activation_name} \nopt_name: {opt_name} \nlearning_rate: {learning_rate}")

                time_in = time.time()
                
                # Initialize the model
                model = MLP(
                    param_dtype=jnp.complex64,
                    hidden_dims=dimensions,
                    activation=activation,
                    )
                # ...with rng keys 
                rng= jax.random.PRNGKey(0)
                key1, key2 = jax.random.split(rng)
                keys = {'params': key1, 'dropout': key2}

                params = model.init(keys['params'], jnp.ones((4*4), dtype=jnp.complex64)) 
                output=model.apply(params, jnp.ones((4*4), dtype=jnp.complex64))

                hi = nk.hilbert.Spin(s=0.5, N = int(np.prod(settings['size'])))

                sampler = nk.sampler.MetropolisLocal(hi, dtype=complex)
                #print(sampler.sample_next(NN_arch, model_params))

                optimizer = opt_name_dict[opt_name](learning_rate=learning_rate)

                vstate = nk.vqs.MCState(sampler, model, n_samples = settings['sampler']['n_samples'])
                is_holo = nk.utils.is_probably_holomorphic(vstate._apply_fun, vstate.parameters, vstate.samples, vstate.model_state)

                SR= nk.optimizer.SR(diag_shift = diag_shift, holomorphic = True)

                # Create the Hamiltonian
                oxa=OxalateJKGamma(settings['size'], 
                                [strength, theta, phi],
                                    bc='periodic', order='default_2')
                H = Runner(oxa.cm).build_hamiltonian()

                gs = nk.VMC(
                    hamiltonian=H,
                    optimizer=optimizer,
                    preconditioner=SR,
                    variational_state=vstate)
                log = nk.logging.RuntimeLog()
                keeper= BestIterKeeper(H, np.prod(settings['size']), baseline = 1e-8,
                    #filename=pathlib.Path("best_state.nk"),
                )
                gs.run(n_iter=n_iter, out=log, callback= [keeper.update])
                time_out = time.time()
                time_exe= time_out - time_in
                
                if ED:
                    E_ED, x_ED = Runner(oxa.cm).exact_energy_lanczos(eigenstates=True)
                    E_ED = E_ED.squeeze(-1)
                    keeper.E_ED = E_ED


                sim_label = f"_{d}_{a}_{opt_name}_{learning_rate}"
                if dump_simulation:
                    # For plotting architecture
                    set_dim=[f"D({dim})" for dim in dimensions] 
                    set_act=[f"A({act})" for act in activation_name]
                    setup='||'
                    for i in range(len(dimensions)):
                        setup+=f" {set_dim[i]} |"
                        setup+=f" {set_act[i]} |" if '0' not in set_act[i] else ''
                        
                    setup+="|"
                    
                    dump_setup ={'size': settings['size'], 'theta': theta,
                                'phi': phi, 'opt_name': opt_name,
                                'learning_rate': learning_rate,
                                'time_exe': time_exe, 'architecture': setup,
                                'sim_label': sim_label
                            }


                    dump_callback(log, dump_setup)

                

                # Extract some results
                vstate = keeper.best_state
                E_best = np.array(keeper.best_energy)
                vscore = np.array(keeper.vscore)
                if ED:
                    error=np.array(np.abs(E_best-E_ED)/np.abs(E_ED))
                    results = np.array([E_best, E_ED, error, vscore])
                else:
                    results = np.array([E_best, vscore])
                # Save the results

                dump_setup ={
                            'size': settings['size'], 
                            'strength': strength,
                            'theta': theta,
                            'phi': phi, 
                            
                            'model': {
                                'name': 'MLP',
                                'dense_dim': str(dimensions),
                                'activation': str(activation_name),
                                
                            },
                            'sampler': {
                                'name': settings['sampler']['type'],
                                'n_samples': settings['sampler']['n_samples'], 
                                'rng': str(vstate.sampler_state.rng )      
                                        },
                            'optimizer': opt_name,
                            'learning_rate': learning_rate,
                            'time_exe': time_exe, 

                        }


                save_results(results, vstate, )
                
