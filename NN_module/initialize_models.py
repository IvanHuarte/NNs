from frozendict import deepfreeze
import flax.linen as nn
from flax import traverse_util
import jax
import jax.numpy as jnp

import NN_module.models
from NN_module.models import REGISTRY

activation_dict = {
    "sigmoid": nn.sigmoid,
    "tanh": nn.tanh,
    "softmax": nn.softmax,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "selu": nn.selu,
    "elu": nn.elu,
    "softplus": nn.softplus,
    "relu": nn.relu,
}


def print_tree(tree, prefix="", values=False):
    for key, val in tree.items():

        if isinstance(val, dict):
            print(prefix + str(key))
            print_tree(val, prefix + "  ", values=values)
        else:
            if values:
                print(prefix + f"{str(key)}: {str(val)}")
            else:
                print(prefix + str(key))


class FactoryBuilder:

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

        setup = self.insert_external_kwargs(setup, extra_args)
        self.setup = deepfreeze(self.preprocess_setup(setup))
        self.model = self.build_module(self.setup, extra_args)

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

    def recursive_list_to_tuple(self, target):

        out = []
        if all(isinstance(element, list) for element in target):
            for element in target:
                out.append(self.recursive_list_to_tuple(element))
        else:
            return tuple(target)
        return tuple(out)

    def insert_external_kwargs(self, setup: dict, external_args: dict):
        """
        To add some arguments (e.g. lattice_size) which are needed but not
        native in NN configurations or changes during simulations.
        """

        latsize_nn = [
            "CNN",
            "CvT",
            "CNNPh",
            "EDPPh",
            "CNNClsf",
            "CNNbinClsf",
            "CvTaps",
            "MLP",
        ]

        # Add lattice_size to modules which natively need spatial info and/or
        # is necesary to perform 2D traslational symmetries
        if "module" in setup.keys() and "setup" in setup.keys():
            if setup["module"] in latsize_nn:
                setup["setup"]["lattice_size"] = external_args["lattice_size"]

        if "symm_2D" in setup:
            setup["lattice_size"] = external_args["lattice_size"]

        for _, val in setup.items():
            if isinstance(val, dict):
                self.insert_external_kwargs(val, external_args)

        return setup

    def preprocess_setup(self, setup: dict) -> dict:

        clean_setup = {}

        for k, v in setup.items():

            if isinstance(v, dict):
                v = self.preprocess_setup(v)

            elif isinstance(v, list):
                v = self.recursive_list_to_tuple(v)

            elif "activation" in k:
                if all([type(act) in [str, int] for act in v]):
                    v = tuple([activation_dict[act] if act != 0 else 0 for act in v])

            clean_setup[k] = v

        return clean_setup

    def build_module(self, setup, extra_args):

        module_name = setup["module"]
        setup = setup["setup"]
        clss = self.get_model_class(module_name)

        symm_Z2 = setup["symm_Z2"] if "symm_Z2" in setup else False
        trivial_Z2 = setup["trivial_Z2"] if "trivial_Z2" in setup else False
        symm_2D = setup["symm_2D"] if "symm_Z2" in setup else False
        lattice_size = setup["lattice_size"] if "symm_2D" in setup else None
        squeeze = jnp.squeeze if "squeeze" in setup else lambda x: x

        if module_name == "SplitTraining":
            modulus = self.build_module(setup["modulus_setup"], extra_args)
            phase = self.build_module(setup["phase_setup"], extra_args)

            return clss(
                ModulusNet=modulus,
                PhaseNet=phase,
                symm_Z2=symm_Z2,
                trivial_Z2=trivial_Z2,
                symm_2D=symm_2D,
                lattice_size=lattice_size,
                squeeze=squeeze,
            )

        elif module_name == "Sequential":
            seq_module = tuple(
                [
                    self.build_module(seq_setup, extra_args)
                    for name, seq_setup in setup.items()
                    if name != "Ending"
                ]
            )
            ending_module = self.build_module(setup["Ending"], extra_args)

            return clss(
                Seq=seq_module,
                End=ending_module,
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

            return clss(
                Trans=trans_module,
                operation=setup["operation"],
                post_norm=setup["post_norm"],
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


# def preprocess_setup(setup: dict) -> dict:

#     clean_setup = {}

#     for k, v in setup.items():

#         if isinstance(v, dict):
#             v = preprocess_setup(v)

#         elif isinstance(v, list):
#             v = recursive_list_to_tuple(v)

#         elif "activation" in k:
#             if all([type(act) in [str, int] for act in v]):
#                 v = tuple([activation_dict[act] if act != 0 else 0 for act in v])

#         clean_setup[k] = v

#     return clean_setup


# def init_model_auto(model_setup):

#     model_class = get_model_class(model_setup["path"])
#     adapted_setup = preprocess_setup(model_setup["setup"])

#     return model_class(**adapted_setup)


# def init_model(name, model_setup, split_training=True):

#     if split_training:

#         model_setup["modulus_setup"]["setup"]["name"] = "modulus"
#         model_setup["phase_setup"]["setup"]["name"] = "phase"

#         model_setup["modulus_setup"]["setup"]["lattice_size"] = model_setup[
#             "lattice_size"
#         ]
#         model_setup["phase_setup"]["setup"]["lattice_size"] = model_setup[
#             "lattice_size"
#         ]

#         lattice_size = (
#             tuple(model_setup["lattice_size"]) if model_setup["symm_2D"] else None
#         )
#         token_size = (
#             tuple(model_setup["modulus_setup"]["setup"]["token_size"])
#             if "ViT" in name
#             else None
#         )

#         # model_setup = preprocess_setup(model_setup)

#         modulus_clss = get_model_class(model_setup["modulus_setup"]["path"])
#         phase_clss = get_model_class(model_setup["phase_setup"]["path"])

#         frozen_modulus_setup = deepfreeze(model_setup["modulus_setup"]["setup"])
#         frozen_phase_setup = deepfreeze(model_setup["phase_setup"]["setup"])

#         return SplitTraining(
#             modulus_clss=modulus_clss,
#             phase_clss=phase_clss,
#             modulus_setup=frozen_modulus_setup,
#             phase_setup=frozen_phase_setup,
#             symm_2D=False,
#             symm_Z2=model_setup["symm_Z2"],
#             trivial_Z2=model_setup["trivial_Z2"],
#             lattice_size=lattice_size,
#             token_size=token_size,
#         )
