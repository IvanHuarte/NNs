import jax
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


def is_subsequence(a, b):
    it = iter(b)
    return all(x in it for x in a)


def mask_branch(mode_paths, path2code):
    def _callable(path, leaf):
        m = next(
            (mod_path for mod_path in mode_paths if is_subsequence(m, path)), "freeze"
        )

    return _callable


def init_mask(path, leaf):
    return "-1"


def masked_optimizer(params, mode_paths, path2code):
    """Genera una máscara universal para `params` basada en el modo.

    Args:
        params: Parámetros del modelo.
        mode: Parte a la que aplicar la máscara ('modulus'/'phase'/'both'/None)
        transform_map: Mapa de transformaciones a aplicar.
    Returns:
        Pytree: Máscara con la misma estructura que `params`.

    """

    init_tree = flax.traverse_util.path_aware_map(init_mask, params)

    trans_tree = flax.traverse_util.path_aware_map(
        mask_branch(mode_paths, path2code), params
    )

    assert jax.tree_util.tree_map(lambda leaf: leaf != "-1", trans_tree).all()

    return trans_tree
