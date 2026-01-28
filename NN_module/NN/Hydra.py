import jax
import jax.numpy as jnp

from NN_module.NN.NN import NeuralNetwork
from NN_module.NN.utils import (
    setup_from_template,
    get_code2path_tree,
    print_architecture,
    get_code2path_flatten,
    change_module_attr,
    greedy_transplant,
    get_subtree,
    set_subtree,
)


class Hydra(NeuralNetwork):
    """
    Dynamic neural architecture manager with evolutionary stages.

    Hydra extends NeuralNetwork by allowing the architecture to evolve
    over multiple stages, where each stage may:
        - Change the network topology.
        - Modify module attributes.
        - Update symmetry constraints.
        - Transplant weights from previous models.

    The object persists across stages while the underlying model
    is reconstructed in-place.
    """

    def __init__(self, hydra_config, **external_kwargs):
        """
        Initialize Hydra with an evolutionary configuration.

        Parameters
        ----------
        hydra_config : dict
            Global Hydra configuration dictionary. Must define:
                - architecture templates
                - evolution stages
                - storage and symmetry rules

        **external_kwargs : dict
            External parameters required by the models
            (e.g. lattice_size, physical constants).
        """

        self.n_stage = 0

        self.print_arch = hydra_config["print_arch"]

        self.storage = hydra_config["storage"]
        self.symm_wrapper = hydra_config["symm_wrapper"]
        self.evolution_name = hydra_config["selection"]
        self.arch_evolution = hydra_config[self.evolution_name]
        self.external_args = external_kwargs

        # Stage 0
        self.template = self.arch_evolution["stage_0"]["template"]
        self.save_params = self.arch_evolution["stage_0"]["save_params"]


        # Build the initial NN model. Adapt first template 'model_0' to a configuration dictionary.
        setup = setup_from_template(self.template, self.storage, self.symm_wrapper)
        self.storage["stage_0"] = setup
        
        super().__init__(setup, **self.external_args)

        if "lattice_size" in self.external_args:
            self.update_info()
        self.params_history = {}

    def setup_from_template(self, template, storage, symm_wrappers):
        return setup_from_template(template, storage, symm_wrappers)

    def get_code2path(self, params):
        return get_code2path_flatten(params)

    def update_info(self, N=None):
        """
        Compute and display architecture metadata.

        Prints:
            - Hierarchical architecture tree.
            - Total number of parameters.
            - Total memory footprint.

        Parameters
        ----------
        N : int, optional
            Input dimension. If not provided, inferred from lattice_size.
        """

        assert (
            "lattice_size" in self.external_args or N != None
        ), f"No size info (N) in internal state 'external_args' nor local "
        if N is None:
            self.N = int(jnp.prod(jnp.array(self.external_args["lattice_size"])))
        else:
            self.N = N

        params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.N)))

        print("********************* ARCHITECTURE INFO *********************\n")
        code2path_tree = get_code2path_tree(params["params"])
        print_architecture(code2path_tree, print_all=self.print_arch)
        print("\n")
        self.n_params, self.nbytes = self.get_params_info(self.model, self.N, True)
        print("\n*************************************************************")

    def arch_evol(self, old_params):
        """
        Advance to the next evolutionary stage.

        This method:
            - Saves current parameters if requested.
            - Loads the next stage configuration.
            - Updates symmetry settings.
            - Rebuilds the neural network architecture.
            - Updates internal metadata.

        Parameters
        ----------
        old_params : pytree
            Parameters of the previous model, used for later transplantation.
        """
        # PREVIOUS EON
        if self.save_params:
            print(f"Saving stage_{self.n_stage} parameters....")
            self.params_history[f"stage_{self.n_stage}"] = old_params
            print("Saved.")

        # NEXT EON
        self.n_stage += 1

        stage_config = self.arch_evolution[f"stage_{self.n_stage}"]
        self.template = stage_config["template"]
        self.save_params = stage_config["save_params"] if "save_params" in stage_config else ""
        self.load = stage_config["load"] if "load" in stage_config else []

        # Update symmetry settings, in the case.
        if "symm_wrapper" in stage_config:
            for k, v in stage_config["symm_wrapper"]:
                assert k in self.symm_wrapper
                self.symm_wrapper[k] = v

        # Create stage setup and change module attributes, in the case
        raw_setup = setup_from_template(self.template, self.storage, self.symm_wrapper)
        if "change_attr" in stage_config:
            if stage_config["change_attr"]:
                raw_setup = change_module_attr(raw_setup, stage_config["change_attr"])

        # Create new NN model
        self.initialize_from_setup(raw_setup, self.external_args)

        # Save the setup
        self.storage[f"stage_{self.n_stage}"] = self.setup

        # Show architecture and update metadata
        self.update_info(self.N)

    def weight_transplantation(self, new_params, return_c2p=True):
        """
        Transplant compatible weights from previous models.

        This method performs block-wise greedy transplantation between
        parameter trees based on precomputed anchor paths.

        For each load instruction:
            1. Extract source and target subtrees.
            2. Match compatible parameters using greedy strategy.
            3. Copy values into the new parameter tree.

        Parameters
        ----------
        new_params : pytree
            Parameters of the newly constructed model.

        code2path : bool, default=True
            If True, also return the updated code-to-path mapping.

        Returns
        -------
        new_params : pytree
            Updated parameter tree with transplanted weights.

        new_c2p : dict (optional)
            Mapping from codes to absolute parameter paths.
        """

        new_c2p = self.get_code2path(new_params)

        for params_name, old_code, new_code in self.load:

            old_params = self.params_history[params_name]
            old_c2p = self.get_code2path(old_params)

            old_path_block = old_c2p[old_code]
            new_path_block = new_c2p[new_code]

            old_params_block = get_subtree(old_params, old_path_block)
            new_params_block = get_subtree(new_params, new_path_block)

            trasplant = greedy_transplant(old_params_block, new_params_block)

            for old_rel, new_rel in trasplant:
                old_abs = old_path_block + old_rel
                new_abs = new_path_block + new_rel

                val = get_subtree(old_params, old_abs)
                new_params = set_subtree(new_params, new_abs, val)

        if return_c2p:
            return new_params, new_c2p
        else:
            return new_params
