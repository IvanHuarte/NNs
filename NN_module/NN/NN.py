from frozendict import deepfreeze
import flax.linen as nn
import jax
import jax.numpy as jnp

from NN_module.NN import REGISTRY
from NN_module.NN.utils import preprocess_setup, insert_external_kwargs
from NN_module.ST_utils import print_tree


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
        self.model = self.build_module(deepfreeze(self.setup), external_args)

    def get_model(self):
        return self.model

    def get_model_class(self, module_name):
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

    def build_module(self, setup, external_args):
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

        module_name = setup["module"]
        setup = setup["setup"]
        clss = self.get_model_class(module_name)

        lattice_size = setup["lattice_size"] if "symm_2D" in setup else None

        symm_Z2 = setup["symm_Z2"] if "symm_Z2" in setup else False
        trivial_Z2 = setup["trivial_Z2"] if "trivial_Z2" in setup else False

        symm_2D = setup["symm_2D"] if "symm_2D" in setup else False
        irrep = setup["irrep"] if "irrep" in setup else (0, 0)
        use_anchor = setup["use_anchor"] if "use_anchor" in setup else False

        squeeze = jnp.squeeze if "squeeze" in setup else lambda x: x

        if module_name == "SplitTraining":
            modulus = self.build_module(setup["modulus"], external_args)
            phase = self.build_module(setup["phase"], external_args)

            return clss(
                ModulusNet=modulus,
                PhaseNet=phase,
                symm_Z2=symm_Z2,
                trivial_Z2=trivial_Z2,
                symm_2D=symm_2D,
                irrep=irrep,
                use_anchor=use_anchor,
                lattice_size=lattice_size,
                squeeze=squeeze,
            )

        elif module_name == "Sequential":
            seq_module = tuple(
                [
                    self.build_module(seq_setup, external_args)
                    for name, seq_setup in setup.items()
                    if name != "ZZ"
                ]
            )
            ZZ_module = self.build_module(setup["ZZ"], external_args)

            return clss(
                Seq=seq_module,
                ZZ=ZZ_module,
                symm_Z2=symm_Z2,
                trivial_Z2=trivial_Z2,
                symm_2D=symm_2D,
                lattice_size=lattice_size,
                squeeze=squeeze,
            )

        elif module_name == "Transversal":
            trans_module = tuple(
                [
                    self.build_module(trans_setup, external_args)
                    for trans_setup in setup.values()
                    if isinstance(trans_setup, dict)
                ]
            )
            operation = setup["operation"] if "operation" in setup else "sum"
            post_norm = setup["post_norm"] if "post_norm" in setup else False

            return clss(
                Trans=trans_module,
                operation=operation,
                post_norm=post_norm,
                symm_Z2=symm_Z2,
                trivial_Z2=trivial_Z2,
                symm_2D=symm_2D,
                lattice_size=lattice_size,
                squeeze=squeeze,
            )

        elif module_name is None:
            return None

        else:
            return clss(**setup)
