import jax
import jax.numpy as jnp

from NN_module.NN import (
    REGISTRY,
    __all_factories__,
    __all_single__,
    __all_symm__,
    factory_submodule_dict,
    factory_submodule_tags,
    symm_submodule_dict,
    undefined_submodule_number,
)
from NN_module.NN.utils import insert_external_kwargs, preprocess_setup
from NN_module.utils import print_tree


class NeuralNetwork:
    """
    Generic neural network builder based on a declarative configuration tree.

    This class constructs an arbitrary Flax neural network architecture
    from a nested configuration dictionary ("setup"), which fully encodes
    the model topology and hyperparameters.

    The class is intentionally *stateless* beyond:
        - self.setup : the processed configuration tree
        - self.model : the instantiated Flax module

    It is designed to be re-initializable with a new setup at any time,
    enabling dynamic architecture changes (e.g. for neural architecture
    search or curriculum learning).
    """

    def __init__(self, setup, **kwargs):
        """
        Build a neural network from a configuration tree.

        Parameters
        ----------
        setup : dict (pytree)
            Nested dictionary encoding the full network architecture.
            Each node must contain a "module" key and a "setup" sub-dictionary.

        **kwargs : dict
            External arguments required by some modules
            (e.g. lattice_size, symmetry flags, etc).
            These are automatically injected into the setup tree.
        """
        # print(kwargs)
        # print(setup)

        self.initialize_from_setup(setup, kwargs)

    def initialize_from_setup(self, setup, external_args):
        """
        (Re)initialize the network from a setup configuration.

        This method fully rebuilds the internal model, and can be safely
        called multiple times on the same object to change the architecture.

        Steps:
            1. Inject external arguments into the setup tree.
            2. Preprocess and normalize the setup structure.
            3. Freeze the setup (hashable & immutable).
            4. Recursively build the Flax module tree.

        Parameters
        ----------
        setup : dict (pytree)
            Configuration tree describing the architecture.

        external_args : dict
            External parameters to be propagated into the setup.
        """

        setup = insert_external_kwargs(setup, external_args)
        self.setup = preprocess_setup(setup)

        print("\nBuilding Neural Network from setup...\n")
        # print_tree(self.setup, values=True)

        self.model = self.build_model(self.setup, external_args)

    def get_model(self):
        return self.model

    def get_model_class(self, module_name):
        try:
            return REGISTRY[module_name]
        except:
            module_name = module_name.split("_")[0]
            return REGISTRY[module_name]

    def get_params_info(self, model, N, show_info=False):
        variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, N)))
        params = variables["params"]
        nbytes = sum(
            x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(params)
        )
        nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
        if show_info:
            print(f"NN stats: {nparams} parameters ({nbytes/(1024**2)} MB)")
        return nparams, nbytes

    def print_setup(self, values=True):
        print_tree(self.setup, values=values)
        print("\n")

    def build_model(self, setup, external_args, depth=0):
        """
        Recursively construct a Flax module from a setup subtree.

        This function interprets the declarative setup format and maps it
        to actual Flax modules using the global REGISTRY.

        Supported high-level structural modules:
            - SplitTraining
            - Sequential
            - Transversal

        Leaf nodes are assumed to be standard Flax modules.

        Parameters
        ----------
        setup : dict
            Single node of the setup tree (must contain "module" and "setup").

        external_args : dict
            External arguments propagated to all modules.

        Returns
        -------
        flax.linen.Module
            Instantiated Flax module corresponding to this subtree.
        """
        if depth == 0:
            print("Getting wrapped model for symmetrization...")
            clss = self.get_model_class("SymmWrapper")
            NN_model = self.build_model(setup, external_args, depth=1)
            print("Model built and wrapped\n\n")
            return clss(NN_model, lattice_size=tuple(external_args["lattice_size"]))

        module_name = setup["module"]
        setup = setup["setup"]
        submodules = {
            k: v
            for k, v in setup.items()
            if any(tag in k for tag in factory_submodule_tags)
        }

        extra_args = {
            k: v
            for k, v in setup.items()
            if k not in (*submodules.keys(), "module", "setup")
        }

        clss = self.get_model_class(module_name)

        if module_name in __all_symm__:
            expected_submodules = symm_submodule_dict[module_name]
            if isinstance(expected_submodules, str):
                expected_submodules = [expected_submodules]
            init_submodules = {
                sub_tag: self.build_model(sub_setup, external_args, depth=depth + 1)
                for sub_tag, sub_setup in zip(expected_submodules, submodules.values())
            }
            return clss(**init_submodules, **extra_args)

        elif module_name in __all_factories__:
            expected_submodules = factory_submodule_dict[module_name]
            if isinstance(expected_submodules, str):
                expected_submodules = [expected_submodules]

            if module_name in undefined_submodule_number:

                init_submodules = {
                    sub_tag: tuple(
                        [
                            self.build_model(sub_setup, external_args, depth=depth + 1)
                            for sub_setup in submodules.values()
                            if isinstance(sub_setup, dict)
                        ]
                    )
                    for sub_tag in expected_submodules
                }

            else:
                init_submodules = {
                    sub_tag: self.build_model(
                        submodules[sub_tag], external_args, depth=depth + 1
                    )
                    for sub_tag in expected_submodules
                }

            return clss(**init_submodules, **extra_args)

        elif module_name in __all_single__:
            return clss(**extra_args)

        elif module_name is None:
            return None
        else:
            raise NotImplementedError(f"Module {module_name} not found in REGISTRY.")
