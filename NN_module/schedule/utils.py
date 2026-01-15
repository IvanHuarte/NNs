import flax


def get_schedule_label(split_training, setup):

    if split_training:
        label = "ST"

        epo_st = setup["epochs_struct"]
        mod_st = setup["modes_struct"]
        epo_st = [period for eon in epo_st for era in eon for period in era]
        mod_st = [period for eon in mod_st for era in eon for period in era]

        for epoch, modes in zip(epo_st, mod_st):
            label += "_"
            label += f"{epoch}{modes}"

    else:
        label = "Both"

    return label


def schedule_from_array(x):
    def schedule_fun(step):
        return x[step]

    return schedule_fun


def make_mask(params, predicate):
    """Genera una máscara con la misma estructura que `params`,
    donde se aplica `predicate(path)` a cada subárbol.
    """

    def apply_mask(tree, path=()):
        if isinstance(tree, dict):
            return {k: apply_mask(v, path + (k,)) for k, v in tree.items()}
        else:
            return predicate(path)

    return apply_mask(params)


def mask_modulus(path, leaf):
    return "freeze" if "ModulusNet" in path else "train_phase"


def mask_phase(path, leaf):
    return "freeze" if "PhaseNet" in path else "train_modulus"


def mask_both(path, leaf):
    return "freeze"


def train_both(path, leaf):
    return (
        "train_modulus"
        if "ModulusNet" in path
        else "train_phase" if "PhaseNet" in path else "train"
    )


def masked_optimizer(params, mode=None):
    """Genera una máscara universal para `params` basada en el modo.

    Args:
        params: Parámetros del modelo.
        mode: Parte a la que aplicar la máscara ('modulus'/'phase'/'both'/None)
        transform_map: Mapa de transformaciones a aplicar.
    Returns:
        Pytree: Máscara con la misma estructura que `params`.

    """

    if mode == "B":
        trans_tree = flax.traverse_util.path_aware_map(train_both, params)
    elif mode == "P":
        trans_tree = flax.traverse_util.path_aware_map(mask_modulus, params)
    elif mode == "M":
        trans_tree = flax.traverse_util.path_aware_map(mask_phase, params)
    elif mode == None:
        trans_tree = flax.traverse_util.path_aware_map(mask_both, params)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'modulus', 'phase' or None (for both)."
        )

    return trans_tree
