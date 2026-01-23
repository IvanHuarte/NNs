import jax
import flax
import flax.linen as nn

from NN_module.NN import (
    EXTERNAL_ARGS, 
    __all_single__, 
    factory_submodule_dict,
    factory_submodule_tags
)
from NN_module.ST_utils import print_tree


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
# Adapt templates in config_Hydra* to 
# NeuralNetwork config dictionary
#############################################



def get_submodules(father, n_mod):
    if father == "Sequential":
        submodules = [f"Seq_{i}" for i in range(n_mod - 1)] + ["End"]
    elif father == "Transversal":
        submodules = [f"Trans_{i}" for i in range(n_mod)]
    elif father == "SplitTraining":
        submodules = ["modulus", "phase"]
    return submodules


def config_from_template(template, storage, symm_wrappers, father="", depth=0):

    config = {}
    if not father:
        for k, v in template.items():
            return config_from_template(v, storage, symm_wrappers, father=k, depth=depth+1)

    elif father in factory_submodule_dict.keys():
        n_mod = len([1 for _, v in template.items() if isinstance(v, dict)])
        submodules = get_submodules(father, n_mod)
        config["module"] = father
        config["setup"] = {}
        for i, (k, v) in enumerate(template.items()):
            if isinstance(v, dict):
                config['setup'].update(**{submodules[i] : config_from_template(v, storage, symm_wrappers, father=k, depth=depth+1)})
            else:
                config['setup'].update(**{k:v})
        if depth == 1:
            config["setup"].update(**symm_wrappers)

    else:
        if isinstance(template, dict):

            if template:

                for k, v in template.items():
                    config["module"] = k
                    config["setup"] = config_from_template(v, storage, symm_wrappers, father=k, depth=depth+1)

            else:
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
        tree[name] = {
            "_idx": idx,
            "_children": get_code2path_tree(subtree, prefix=idx)
        }
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
    if "_" in raw_name: raw_name = raw_name.split("_")[0]
    if "Worker" in raw_name: raw_name = raw_name.split("Worker")[0]
    if "Tokenize" in raw_name: raw_name = raw_name.split("Tokenize")[0]

    if raw_name in __all_single__:
        return f"({raw_name})"
    else:
        return ''
    
def print_architecture(tree, print_all=True, prefix="", is_last=True):

    keys = list(tree.keys())

    for i, name in enumerate(keys):
        data = tree[name]
        last = (i == len(keys) - 1)

        is_main = is_main_module(name)
        children = list(data['_children'].keys())
        simple_module = is_simple_module(children) if len(children) == 1 else False
        print(simple_module)

        if not print_all and not is_main:
            continue

        annotation = ''
        if is_main and simple_module:
            annotation += f"{simple_module}"

        connector = "└── " if last else "├── "
        print(prefix + connector + f"{name} {annotation}  --->  ({data['_idx']})")

        if data['_children']:
            new_prefix = prefix + ("    " if last else "│   ")
            print_architecture(
                data['_children'],
                print_all,
                prefix=new_prefix,
                is_last=last
            )




