from frozendict import deepfreeze
import flax.linen as nn
import jax
import jax.numpy as jnp

from NN_module.NN import REGISTRY
from NN_module.NN.utils import preprocess_setup, insert_external_kwargs
from NN_module.ST_utils import print_tree


class NeuralNetwork():

    def __init__(self, setup, **kwargs):
        """
        Class for automatically build an arbitry network architecture
        based only on a configuration dictionary. It can use either
        simple modules and factories, being the latests flux information
        organizers.
        __init__:
            -setup:(dict) Pytree dictionary which contains all NN
                    architecture information.
            -kwargs:(dict) External arguments needed by modules

        First, it inserts recursively the kwargs in required setup sites
        via `insert_external_kwargs`. Next, it performs a prepocessing in
        the setup data (`preprocess_setup`) and then `deepfreeze` it to
        build an inmmutable and hashable dictionary. Finally, it builds
        the NN architecture via `build_module`
        """

        extra_args = kwargs

        setup = insert_external_kwargs(setup, extra_args)
        self.setup = preprocess_setup(setup)
        self.model = self.build_module(deepfreeze(self.setup), extra_args)

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

    def build_module(self, setup, extra_args):

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
            modulus = self.build_module(setup["modulus"], extra_args)
            phase = self.build_module(setup["phase"], extra_args)

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
                    self.build_module(seq_setup, extra_args)
                    for name, seq_setup in setup.items()
                    if name != "End"
                ]
            )
            end_module = self.build_module(setup["End"], extra_args)

            return clss(
                Seq=seq_module,
                End=end_module,
                symm_Z2=symm_Z2,
                trivial_Z2=trivial_Z2,
                symm_2D=symm_2D,
                lattice_size=lattice_size,
                squeeze=squeeze,
            )

        elif module_name == "Transversal":
            trans_module = tuple(
                [
                    self.build_module(trans_setup, extra_args)
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
