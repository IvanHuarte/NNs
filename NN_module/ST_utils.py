import jax
import flax
import optax

def make_mask(params, predicate):
    """Genera una máscara con la misma estructura que `params`,
    donde se aplica `predicate(path)` a cada subárbol.
    """
    def apply_mask(tree, path=()):
        if isinstance(tree, dict):
            return {
                k: apply_mask(v, path + (k,))
                for k, v in tree.items()
            }
        else:
            return predicate(path)
    return apply_mask(params)



def mask_modulus(path, leaf):
    return 'freeze' if path[0] == 'modulus' else 'train'

def mask_phase(path, leaf):
    return 'freeze' if path[0] == 'phase' else 'train'

def mask_both(path, leaf):
    return 'freeze'

def masked_optimizer(params, transform_map, mode=None):
    """Genera una máscara universal para `params` basada en el modo.
    
    Args:
        params: Parámetros del modelo.
        mode: Parte a la que aplicar la máscara ('modulus','phase')
        transform_map: Mapa de transformaciones a aplicar.
    Returns:
        dict: Máscara con la misma estructura que `params`.
    """

    if mode == 'modulus':
        trans_tree = flax.traverse_util.path_aware_map(mask_modulus, params)
    elif mode == 'phase':
        trans_tree = flax.traverse_util.path_aware_map(mask_phase, params)
    elif mode == 'both':
        trans_tree = flax.traverse_util.path_aware_map(mask_both, params)
    elif mode is None:
        return transform_map['train']
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'modulus', 'phase' or None (for both).")
    
    optimizer = optax.multi_transform(transform_map, trans_tree)
    
    return optimizer


def compare_params(old_params, new_params, atol=1e-12):
    def compare_fn(p_old, p_new):
        return not jax.numpy.allclose(p_old, p_new, atol=atol)

    diffs = jax.tree_util.tree_map(compare_fn, old_params, new_params)
    return diffs