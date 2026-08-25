import json

import flax.linen as nn
import jax.numpy as jnp
import netket as nk
import orbax.checkpoint as ocp

from NN_module.NN import (
    EXTERNAL_ARGS,
    __all_single__,
    factory_submodule_dict,
    factory_submodule_tags,
    get_submodules,
    symm_submodule_dict,
)


class ModReLU(nn.Module):
    bias_init: float = 0.0

    @nn.compact
    def __call__(self, z):
        b = self.param("b", lambda key: jnp.array(self.bias_init))
        r = jnp.abs(z) + b
        # ReLU on modulus
        m = jnp.maximum(r, 0)
        return m * (z / jnp.abs(z))


def cardioid(z, eps=1e-8):
    r = jnp.abs(z) + eps
    scale = 0.5 * (1.0 + jnp.real(z) / r)
    return scale * z


def make_split(activation):
    def nruter(x):
        return activation(x.real) + 1.0j * activation(x.imag)

    return nruter


activation_dict = {
    "relu": nn.relu,
    "logcosh": nk.nn.activation.log_cosh,
    "sigmoid": nn.sigmoid,
    "tanh": nn.tanh,
    "softmax": nn.softmax,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "selu": nn.selu,
    "elu": nn.elu,
    "softplus": nn.softplus,
    "modrelu": ModReLU,
    "cardioid": cardioid,
    "Crelu": make_split(nn.relu),
    "Clogcosh": make_split(nk.nn.activation.log_cosh),
    "Csigmoid": make_split(nn.sigmoid),
    "Ctanh": make_split(nn.tanh),
    "Csoftmax": make_split(nn.softmax),
    "Cgelu": make_split(nn.gelu),
    "Cswish": make_split(nn.swish),
    "Celu": make_split(nn.elu),
    "Csoftplus": make_split(nn.softplus),
}


def get_subtree(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def set_subtree(tree, path, new_subtree):
    if len(path) == 0:
        return new_subtree
    key = path[0]
    return {**tree, key: set_subtree(tree[key], path[1:], new_subtree)}


######################################################
#                                                    #
#                   NEURALNETWORK                    #
#                                                    #
######################################################


def recursive_list_to_tuple(target):

    out = []
    if all(isinstance(element, list) for element in target):
        for element in target:
            out.append(recursive_list_to_tuple(element))
    else:
        return tuple(target)
    return tuple(out)


def recursive_tuple_to_list(target):

    out = []
    if all(isinstance(element, tuple) for element in target):
        for element in target:
            out.append(recursive_tuple_to_list(element))
    else:
        return list(target)
    return out


def preprocess_setup(setup: dict) -> dict:

    clean_setup = {}

    for k, v in setup.items():

        if isinstance(v, dict):
            v = preprocess_setup(v)

        elif isinstance(v, list):
            v = recursive_list_to_tuple(v)

        if "activation" in k:

            if all([type(act) in [str, int] for act in v]):
                v = tuple([activation_dict[act] if act != 0 else 0 for act in v])

        clean_setup[k] = v

    return clean_setup


def make_setup_serializable(setup: dict) -> dict:

    clean_setup = {}

    for k, v in setup.items():

        if isinstance(v, dict):
            v = make_setup_serializable(v)

        elif isinstance(v, tuple):
            v = recursive_tuple_to_list(v)

        clean_setup[k] = v

    return clean_setup


def insert_external_kwargs(setup: dict, external_args: dict):
    """
    To add some arguments (e.g. lattice_size) which are needed but not
    native in NN configurations or changes during simulations.
    """

    # Add lattice_size to modules which natively need spatial info and/or
    # is necesary to perform 2D traslational symmetries
    if "module" in setup and "setup" in setup:
        for k, v in external_args.items():

            if setup["module"] in EXTERNAL_ARGS[k]:
                setup["setup"][k] = v

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
# Adapt templates in config_Hydra* to
# NeuralNetwork config dictionary
#############################################


def recursive_build_NN(template, storage, father="", depth=0):

    config = {}
    if not father:
        for k, v in template.items():
            return recursive_build_NN(v, storage, father=k, depth=depth + 1)

    elif father in factory_submodule_dict.keys():
        n_mod = len([1 for _, v in template.items() if isinstance(v, dict)])
        submodules = get_submodules(father, n_mod)
        config["module"] = father
        config["setup"] = {}
        for i, (k, v) in enumerate(template.items()):

            if isinstance(v, dict):
                config["setup"].update(
                    **{
                        submodules[i]: recursive_build_NN(
                            v, storage, father=k, depth=depth + 1
                        )
                    }
                )
            else:
                config["setup"].update(**{k: v})

    else:
        if isinstance(template, dict):

            if template:

                for k, v in template.items():
                    config["module"] = k
                    config["setup"] = recursive_build_NN(
                        v, storage, father=k, depth=depth + 1
                    )

            else:
                if "stage" in father:
                    config["module"] = storage[father]["module"]
                    config["setup"] = storage[father]["setup"]
                else:
                    father = father if "_" not in father else father.split("_")[0]
                    config["module"] = father
                    config["setup"] = storage[father]

        else:
            config[father] = template

    return config


def recursive_build_symm_wrapper(template, storage, symm_wrapper, count=0):

    if count == len(symm_wrapper.keys()):
        return recursive_build_NN(template, storage)

    symm_key = list(symm_wrapper.keys())[count]
    symmetrization = symm_wrapper[symm_key]

    if symmetrization["on"]:
        config = {}
        config["module"] = symm_key
        config["setup"] = {}
        config["setup"]["Wrap"] = recursive_build_symm_wrapper(
            template, storage, symm_wrapper, count=count + 1
        )
        for k, v in symmetrization.items():
            if k != "on":
                config["setup"][k] = v

    else:
        config = recursive_build_symm_wrapper(
            template, storage, symm_wrapper, count=count + 1
        )

    return config


def setup_from_template(template, storage, symm_wrappers):

    config = recursive_build_symm_wrapper(template, storage, symm_wrappers)

    return config


def tree_delete_attributes(tree, attributes, depth=0):
    """
    Deletes attributes from a nested dictionary (tree) structure.

    Parameters
    ----------
    tree : dict
        The nested dictionary from which attributes will be deleted.
    attributes : list
        List of attribute names (keys) to be deleted from the tree.

    Returns
    -------
    dict
        The modified tree with specified attributes removed.
    """
    if not isinstance(tree, dict):
        return tree

    if depth > 2:
        return {
            k: tree_delete_attributes(v, attributes, depth=depth + 1)
            for k, v in tree.items()
            if k not in attributes
        }
    else:
        return {
            k: tree_delete_attributes(v, attributes, depth=depth + 1)
            for k, v in tree.items()
        }


###########################################
# Encode the architecture into int-strings,
# used to modify it in subsequent calls
###########################################
"""

                                    Architecture: (nodes)    
        
                                             X_in
        
                                              ║                                                                     
                                    ╔═════════╩═════════╗                                                     
                              ModulusNet(0)         PhaseNet(1)                                                   
                                    ║                   ║                                                            
         < Transversal > --> ╔══════╩══════╗        ╔═══╩═══╗                                                          
                         ╔═══╩═══╗     ╔═══╩═══╗    ║  CvT  ║ (1)                       
                         ║  CNN  ║     ║  MLP  ║    ╚═══╦═══╝                                    
                         ╚═══╦═══╝     ╚═══╦═══╝        ║                                      
                       (00)  ║             ║ (01)       ║                     
                             ╚══════╦══════╝            ║
                                    ╚═════════╦═════════╝
                                              ║
        
                                             X_out 
                                                                                    
get_code_dict:{                get_code2path:{                        get_code2path_tree:{                    
  "ModulusNet:{                    "0": ("ModulusNet",),                   "ModulusNet": {                         
    "Trans_0:{                     "1": ("PhaseNet",),                        "_idx": "0",                              
        .... {                     "00": ("ModulusNet", Trans_0),             "_children": {                                  
        param0: '00...0',          "01": ("ModulusNet", Trans_1),               "Trans_0": {                                
        param1: '00...1',          ....                                           "_idx": "00",                                    
        ....    "                  inner params                                   "_children": {                             
     ......                                                                       ...... 
}                              }                                      }
                                    
"""


def get_leafcode_dict(node, idx=""):
    """
    Recursivelly builds a dictionary homomorphic with "node", usually
    the nerwork parameters where each leaf has its index-code.

    -module:
    ---param0: 00010
    ---param1: 00011

    Returns: PytreeDict
    """
    nruter = {}

    subnodes_names = node.node_data()[1]
    subnodes = node.children()

    for i, (k, v) in enumerate(zip(subnodes_names, subnodes)):

        if k in symm_submodule_dict.values():
            next_idx = idx
        else:
            next_idx = idx + str(i)

        if not v.children():
            nruter[k] = next_idx

        else:
            nruter[k] = get_leafcode_dict(v, next_idx)

    return nruter


def get_code2path_flatten(params):
    """
    Using `get_code_dict` builds a linear dictionary and each element
    is 'index_code': (path), where the index-code is a int-string which
    codifies a "leaf" or a "node" within the parameters Pytree. Example:

    """

    symm_submodule_labels = list(symm_submodule_dict.values())

    code_dict = {}

    def traverse(node, path=(), code=""):

        if isinstance(node, dict):
            for i, (k, v) in enumerate(node.items()):

                next_code = code if k in symm_submodule_labels else code + str(i)

                traverse(v, path + (k,), next_code)

        else:
            code_dict[code] = path

    traverse(params)

    code_to_path = {}

    for k, v in code_dict.items():
        n_nodes = len(k)
        for i in range(len(k)):
            code_to_path[k[: -n_nodes + i + 1]] = v[: -n_nodes + i + 1]

    code_to_path.pop("")

    return code_to_path


def get_code2path_tree(params, prefix=""):
    """
    Recursively builds a tree from a Flax/JAX pytree of parameters,
    assigning a unique positional index to each node.

    Each node is stored as:
        tree[name] = {
            "_idx": <hierarchical code>,
            "_children": { ... }
        }
    Indices are generated by concatenating child positions at each level:
        "0", "01", "010", etc.
    """
    if not isinstance(params, dict):
        # Hoja, retornamos vacío
        return {}

    tree = {}
    for i, (name, subtree) in enumerate(params.items()):
        if name in symm_submodule_dict.values():
            idx = f"{prefix}"
        else:
            idx = f"{prefix}{i}"
        tree[name] = {"_idx": idx, "_children": get_code2path_tree(subtree, prefix=idx)}
    return tree


##########################################
# PLOT ARCHITECTURE WITH CODE INDICES
##########################################


def is_main_module(name):
    if "_" in name:
        name = name.split("_")[0]
    return name in factory_submodule_tags


def is_simple_module(raw_name):
    raw_name = raw_name[0]
    if "_" in raw_name:
        raw_name = raw_name.split("_")[0]
    if "Worker" in raw_name:
        raw_name = raw_name.split("Worker")[0]
    if "Block" in raw_name:
        raw_name = raw_name.split("Block")[0]

    if raw_name in __all_single__:
        return f"({raw_name})"
    else:
        return ""


def is_wrap(name):
    return name in symm_submodule_dict.values()


def print_architecture(tree, print_all=True, prefix="", is_last=True, wraps=[]):

    keys = list(tree.keys())

    for i, name in enumerate(keys):
        data = tree[name]
        last = i == len(keys) - 1

        iswrap = is_wrap(name)

        if iswrap:
            # Is Symm_Wrap

            children = list(data["_children"].keys())
            if len(children) == 1:
                child_name = children[0]
                if is_wrap(child_name):
                    print_architecture(
                        data["_children"], print_all, prefix, is_last, wraps + [name]
                    )
                else:
                    wraps.append(name)
                    wrap_label = (
                        " ".join(wraps[1:])
                        if len(wraps) > 1
                        else print("No symmetrization")
                    )
                    (
                        print(f"SymmModel: ({wrap_label})")
                        if len(wraps) > 1
                        else print("SymmModel:")
                    )
                    print_architecture(data["_children"], print_all, prefix, is_last)
            else:
                wraps.append(name)
                wrap_label = (
                    " ".join(wraps[1:])
                    if len(wraps) > 1
                    else print("No symmetrization")
                )
                (
                    print(f"SymmModel: ({wrap_label})")
                    if len(wraps) > 1
                    else print("SymmModel:")
                )
                print_architecture(data["_children"], print_all, prefix, is_last)

        else:

            # Is factory
            is_main = is_main_module(name)
            children = list(data["_children"].keys())
            simple_module = ""
            if is_main:
                if len(children) == 1:
                    simple_module = is_simple_module(children)
                else:
                    candidates = [is_simple_module([child]) for child in children]
                    simple_module = next((c for c in candidates if c != ""), "")

            if not print_all and not is_main:
                continue

            annotation = ""
            if is_main and simple_module:
                annotation += f"{simple_module}"

            connector = "└── " if last else "├── "
            print(prefix + connector + f"{name} {annotation}  --->  ({data['_idx']})")

            if data["_children"] and not iswrap:
                new_prefix = prefix + ("    " if last else "│   ")
                print_architecture(
                    data["_children"], print_all, prefix=new_prefix, is_last=last
                )


##########################################
# CHANGE MODULE ATTRIBUTES ON THE FLY
##########################################


def change_values(module_setup, changes):
    for attr, value in changes.items():
        if attr != "_count":
            module_setup[attr] = value
    return module_setup


def change_module_attr(setup, changes):
    """

    :param setup: setup from which the changes are made.
    :param changes: Pytree of structure:
    {"module_name": {
        "_count":(optional) int. If 2 modules in architecture setup
                    have the same name. The changes will be applied to the
                    #_count module.
        "attr_0": new_value_0,
        "attr_1": new_value_1,
        ...
        },
    ...
    }
    """

    new_setup = {}
    if all([key in ["module", "setup"] for key in setup.keys()]):
        new_setup["module"] = setup["module"]
        if setup["module"] in changes.keys():  # Then is a change_target

            if "_count" in changes[setup["module"]]:
                if changes[setup["module"]]["_count"] != 0:
                    changes[setup["module"]]["_count"] += -1
                    new_setup["setup"] = setup["setup"]

                else:
                    new_setup["setup"] = change_values(
                        setup["setup"], changes[setup["module"]]
                    )

            else:
                new_setup["setup"] = change_values(
                    setup["setup"], changes[setup["module"]]
                )

        elif setup["module"] in __all_single__:  # Then is another simple module
            new_setup["setup"] = setup["setup"]

        else:
            new_setup["setup"], changes = change_module_attr(setup["setup"], changes)

    else:  # module, phase, Seq, Trans
        for k, v in setup.items():
            if isinstance(v, dict):
                new_setup[k], changes = change_module_attr(v, changes)
            else:
                new_setup[k] = v

    return new_setup, changes


##########################################
# WEIGHTS TRASPLANTATION OF TRAINED MODELS
##########################################


def flatten_with_paths(tree):
    out = []

    def visit(t, path):
        if isinstance(t, dict):
            for k, v in t.items():
                visit(v, path + (k,))
        elif isinstance(t, (list, tuple)):
            for i, v in enumerate(t):
                visit(v, path + (i,))
        else:
            out.append((path, t))

    visit(tree, ())
    return out


def find_leaf_matches(p0, p1):
    flat0 = flatten_with_paths(p0)
    flat1 = flatten_with_paths(p1)

    matches = []
    for path0, x in flat0:
        for path1, y in flat1:
            if x.shape == y.shape and x.dtype == y.dtype:
                matches.append((path0, path1))
    return matches


def greedy_transplant(old, new):
    matches = find_leaf_matches(old, new)
    used_old = set()
    used_new = set()
    mapping = []

    for p0, p1 in matches:
        if p0 not in used_old and p1 not in used_new:
            mapping.append((p0, p1))
            used_old.add(p0)
            used_new.add(p1)
    return mapping


##########################################
# LOAD PARAMETERS FROM FILE FOR FIRST STAGE
##########################################


def load_pytree(path):
    cp = ocp.PyTreeCheckpointer()
    return cp.restore(path)


def load_from_artifact(artifact_path, checkpoint=0, tag="parameters"):

    with open(artifact_path, "r") as f:
        artifact = json.load(f)

    if checkpoint:
        tag = "checkpoint"

    path = artifact["_artifacts"][tag]
    data = load_pytree(path)

    return data


def load_from_file(path, checkpoint=0, tag="parameters"):

    if ".json" in path:  # Best results
        data = load_from_artifact(path, checkpoint=checkpoint, tag=tag)
        if isinstance(data, str):
            data = load_from_file(data, checkpoint=checkpoint, tag=tag)

    elif ".orbax" in path:  # Direct files
        path += "/" if not path.endswith("/") else ""
        if "checkpoint" in path:
            path += str(checkpoint) + "/" + tag

        data = load_pytree(path)

    return data
