import jax
import jax.numpy as jnp

from NN_module.NN.NN import NeuralNetwork
from NN_module.NN.utils import config_from_template, get_code2path_tree, print_architecture, get_code2path_flatten

class Hydra(NeuralNetwork):

    def __init__(self, hydra_config, **external_kwargs):

        self.print_arch = hydra_config["print_arch"]

        self.storage = hydra_config["storage"]
        self.symm_wrappers = hydra_config["symm_wrappers"]
        self.evolution_name = hydra_config["selection"]
        self.arch_evolution = hydra_config[self.evolution_name]
        self.external_args = external_kwargs 
        self.current_template = self.arch_evolution["stage_0"]

        # Build the initial NN model. Adapt first template 'model_0' to a configuration dictionary.
        config = self.config_from_template(self.current_template, self.storage, self.symm_wrappers)
        super().__init__(config, **self.external_args)

        if 'lattice_size' in self.external_args:
            self.update_info()
        self.params_history = []


    def config_from_template(self, template, storage, symm_wrappers):
        return config_from_template(template, storage, symm_wrappers)
    
    def get_code2path(self, params):
        return get_code2path_flatten(params)

    def update_info(self, N=None):
        assert 'lattice_size' in self.external_args or N != None, (
            f"No size info (N) in internal state 'external_args' nor local "
        )
        if N is None:
            N = int(jnp.prod(jnp.array(self.external_args['lattice_size'])))

        params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, N)))

        print("********************* ARCHITECTURE INFO *********************\n")
        code2path_tree = get_code2path_tree(params)
        print_architecture(code2path_tree, print_all=self.print_arch)

        self.n_params, self.nbytes = self.get_params_info(self.model, N, True)
        print("\n*************************************************************")

    

    
        