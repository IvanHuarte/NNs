import jax
import flax
import optax


class WrappedModel:
    def __init__(self, model, train_modulus, train_phase):
        self.model = model
        self.train_modulus = train_modulus
        self.train_phase = train_phase

    def init(self, rng, x):
        return self.model.init(rng, x,
                               train_modulus=self.train_modulus,
                               train_phase=self.train_phase)

    def apply(self, vars, x, **kwargs):
        return self.model.apply(vars, x,
                                train_modulus=self.train_modulus,
                                train_phase=self.train_phase,
                                **kwargs)

def init_function(model, modulus=True, phase=True):
    
    def core_init(rng, dummy_x, **kwargs):
        return model.init(rng, dummy_x, train_modulus=modulus, train_phase=phase, **kwargs)
    return core_init

def apply_function(model, modulus=True, phase=True):

    def core_apply(variables, x, **kwargs):
        return model.apply(variables, x, train_modulus=modulus, train_phase=phase, **kwargs)
    return core_apply


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
    return 'freeze' if path[0] == 'BatchedSpinViT_0' else 'train'

def mask_phase(path, leaf):
    return 'freeze' if path[0] in ('BatchedMultiLayerPerceptron_0', 'CNN_0') else 'train'

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