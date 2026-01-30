import json
import jax
import flax
import flax.linen as nn
import orbax.checkpoint as ocp

from NN_module.NN import (
    EXTERNAL_ARGS,
    __all_single__,
    factory_submodule_dict,
    factory_submodule_tags,
)


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

        elif "activation" in k:
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
# Adapt templates in config_Hydra* to
# NeuralNetwork config dictionary
#############################################


def get_submodules(father, n_mod):
    if father == "Sequential":
        submodules = [f"Seq_{i}" for i in range(n_mod - 1)] + ["ZZ"]
    elif father == "Transversal":
        submodules = [f"Trans_{i}" for i in range(n_mod)]
    elif father == "SplitTraining":
        submodules = ["modulus", "phase"]
    elif father == "SingleModule":
        submodules = ["single"]
    return submodules


def setup_from_template(template, storage, symm_wrappers, father="", depth=0):

    config = {}
    if not father:
        for k, v in template.items():
            return setup_from_template(
                v, storage, symm_wrappers, father=k, depth=depth + 1
            )

    elif father in factory_submodule_dict.keys():
        n_mod = len([1 for _, v in template.items() if isinstance(v, dict)])
        submodules = get_submodules(father, n_mod)
        config["module"] = father
        config["setup"] = {}
        for i, (k, v) in enumerate(template.items()):
            if isinstance(v, dict):
                config["setup"].update(
                    **{
                        submodules[i]: setup_from_template(
                            v, storage, symm_wrappers, father=k, depth=depth + 1
                        )
                    }
                )
            else:
                config["setup"].update(**{k: v})
        if depth == 1:
            config["setup"].update(**symm_wrappers)

    else:
        if isinstance(template, dict):

            if template:

                for k, v in template.items():
                    config["module"] = k
                    config["setup"] = setup_from_template(
                        v, storage, symm_wrappers, father=k, depth=depth + 1
                    )

            else:
                father = father if "_" not in father else father.split("_")[0]
                config["module"] = father
                config["setup"] = storage[father]

        else:
            config[father] = template

    return config


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

        if not v.children():
            nruter[k] = idx + str(i)

        else:
            nruter[k] = get_leafcode_dict(v, idx + str(i))

    return nruter


def get_code2path_flatten(params):
    """
    Using `get_code_dict` builds a linear dictionary and each element
    is 'index_code': (path), where the index-code is a int-string which
    codifies a "leaf" or a "node" within the parameters Pytree. Example:

    """

    struct = jax.tree_util.tree_structure(params)

    code_dict = get_leafcode_dict(struct)

    code_dict = flax.traverse_util.flatten_dict(code_dict)

    tmp = {}

    for k, v in code_dict.items():
        for i in range(len(k)):
            tmp[k[: i + 1]] = v[: i + 1]

    code_to_path = {}
    for k, v in tmp.items():
        code_to_path[v] = k

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
    if "Tokenize" in raw_name:
        raw_name = raw_name.split("Tokenize")[0]

    if raw_name in __all_single__:
        return f"({raw_name})"
    else:
        return ""


def print_architecture(tree, print_all=True, prefix="", is_last=True):

    keys = list(tree.keys())

    for i, name in enumerate(keys):
        data = tree[name]
        last = i == len(keys) - 1

        is_main = is_main_module(name)
        children = list(data["_children"].keys())
        simple_module = is_simple_module(children) if len(children) == 1 else False

        if not print_all and not is_main:
            continue

        annotation = ""
        if is_main and simple_module:
            annotation += f"{simple_module}"

        connector = "└── " if last else "├── "
        print(prefix + connector + f"{name} {annotation}  --->  ({data['_idx']})")

        if data["_children"]:
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


def load_params_from_file(artifact_path):

    with open(artifact_path, "r") as f:
        artifact = json.load(f)

    path = artifact["_artifacts"]["parameters"]
    parameters = load_pytree(path)

    return parameters
