import json
from collections import defaultdict
from functools import cache

# ---------- Lectura de información ----------


@cache
def load_artifact(path):
    """Carga el json una única vez."""
    with open(path) as f:
        return json.load(f)


def get_irrep_value(tree, criterion):
    """
    Devuelve el valor en tupla de la irrep correspondiente.
    """

    irrep_names = criterion.split("_")
    irrep_value = []
    for symmetry in irrep_names:

        if symmetry == "Tx":
            irrep = find_value(tree["NN"], "irrep")
            irrep_value.append(irrep[0])

        elif symmetry == "Ty":
            irrep = find_value(tree["NN"], "irrep")
            irrep_value.append(irrep[1])

        elif symmetry == "Z2":
            irrep_bool = find_value(tree["NN"], "trivial_Z2")
            irrep = 0 if irrep_bool else 1
            irrep_value.append(irrep)

        else:
            raise NotImplementedError(f"Symmetry {symmetry} not implemented.")

    return tuple(irrep_value)


def find_value(tree, criterion):
    """
    Busca recursivamente criterion en el pytree.

    criterion puede ser una clave simple, por ejemplo:
        "J1"

    o una ruta relativa:
        "model_params/J1"
        "SIM/model_params/J1"

    Devuelve el valor encontrado.
    En el caso de clasificar por irreps, devuelve la tupla correspondiente a la irrep.
    Para ello, criterion debe ser de la forma "irreps_<irrep_name_0>_<irrep_name_1>_..._<irrep_name_n>".
    """

    if "irreps" in criterion:
        return get_irrep_value(tree, criterion.split("_", 1)[1])

    parts = criterion.split("/")

    def search(node):

        if isinstance(node, dict):

            # Intentamos encontrar la ruta completa desde este nodo
            current = node

            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    break
                current = current[part]
            else:
                return current

            # Si no estaba aquí, seguimos buscando recursivamente
            for value in node.values():
                result = search(value)

                if result is not None:
                    return result

        elif isinstance(node, list):

            for value in node:
                result = search(value)

                if result is not None:
                    return result

        return None

    result = search(tree)

    if result is None:
        raise KeyError(f"No se encontró '{criterion}' en el pytree")

    return result


def get_value(path, criterion):
    """
    Obtiene el valor de criterion directamente del JSON.
    """

    artifact = load_artifact(path)

    return find_value(artifact, criterion)


# ---------- Agrupación ----------


def group_by(paths, criterion):
    """
    Agrupa una lista de archivos según criterion.

    criterion:
        "J1"       -> agrupa por J1
        "J1-J2"    -> agrupa por (J1, J2)
    """

    groups = defaultdict(list)

    criteria = criterion.split("-")

    for path in paths:

        if len(criteria) == 1:
            key = get_value(path, criteria[0])
        else:
            key = tuple(get_value(path, c) for c in criteria)

        groups[str(key)].append(path)

    try:
        keys = sorted(groups)
    except TypeError:
        keys = groups.keys()

    return {k: groups[k] for k in keys}


# ---------- Clasificación recursiva ----------


def classify(paths, mode):

    if mode == "":
        return paths

    if "|" in mode:
        criterion, remaining = mode.split("|", 1)
    else:
        criterion, remaining = mode, ""

    groups = group_by(paths, criterion)

    return {key: classify(group, remaining) for key, group in groups.items()}

# -------------- Print dictionary --------------

def print_tree(tree, prefix="", values=True):
    for key, val in tree.items():

        if isinstance(val, dict):
            print(prefix + str(key))
            print_tree(val, prefix + "   ", values=values)
        else:
            if values:
                print(prefix + f"{key!s}: {val!s}")
            else:
                print(prefix + str(key))


# ------------- PLOTTING -------------- #

param2latex = {
    "J": r"J",
    "J1": r"J_1",
    "J2": r"J_2",
    "size": r"size",

}


def join_modes(modes, values):

    assert len(modes) == len(values)
    values = [f"{v:g}" if not isinstance(v, str) else v for v in values]
    equals = [rf"{param2latex[m]}\;=\;{v}" for m, v in zip(modes, values)]
    return r"\;\;".join(equals)


def join_static_modes(static_modes):

    static_modes = [
        (m, f"{v:g}") if not isinstance(v, str) else (m, v) for m, v in static_modes
    ]
    equals = [rf"${param2latex[m]}\;=\;{v}$" for m, v in static_modes]
    return "\n".join(equals)
