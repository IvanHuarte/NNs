import flax.linen as nn
from NN_module.NN import EXTERNAL_ARGS


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


def recursive_list_to_tuple(target):

    out = []
    if all(isinstance(element, list) for element in target):
        for element in target:
            out.append(recursive_list_to_tuple(element))
    else:
        return tuple(target)
    return tuple(out)


def preprocess_setup(setup: dict) -> dict:

    clean_setup = {}

    for k, v in setup.items():

        if isinstance(v, dict):
            v = preprocess_setup(v)

        elif isinstance(v, list):
            v = recursive_list_to_tuple(v)

        elif "activation" in k:
            if all([type(act) in [str, int] for act in v]):
                v = tuple([activation_dict[act] if act != 0 else 0 for act in v])

        clean_setup[k] = v

    return clean_setup


def insert_external_kwargs(setup: dict, external_args: dict):
    """
    To add some arguments (e.g. lattice_size) which are needed but not
    native in NN configurations or changes during simulations.
    """

    # Add lattice_size to modules which natively need spatial info and/or
    # is necesary to perform 2D traslational symmetries
    if "module" in setup.keys() and "setup" in setup.keys():
        for k, v in external_args.items():
            if setup["module"] in EXTERNAL_ARGS[k]:
                setup["setup"][k] = v

    if "symm_2D" in setup:
        setup["lattice_size"] = external_args["lattice_size"]

    for _, val in setup.items():
        if isinstance(val, dict):
            insert_external_kwargs(val, external_args)

    return setup


######################################################
#                                                    #
#                      HYDRA                         #
#                                                    #
######################################################

#############################################
#
#############################################
