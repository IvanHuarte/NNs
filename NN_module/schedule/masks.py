import flax


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

def mask_branch(modes):
    def _callable(path, leaf):
        return next((m for m in modes if m in path), 'freeze')
    
    return _callable


def masked_optimizer(params, modes):
    """Genera una máscara universal para `params` basada en el modo.

    Args:
        params: Parámetros del modelo.
        mode: Parte a la que aplicar la máscara ('modulus'/'phase'/'both'/None)
        transform_map: Mapa de transformaciones a aplicar.
    Returns:
        Pytree: Máscara con la misma estructura que `params`.

    """

    trans_tree = flax.traverse_util.path_aware_map(mask_branch(modes), params)

    return trans_tree
