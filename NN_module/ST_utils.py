import jax
import jax.numpy as jnp
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
    return 'freeze' if 'modulus' in path else 'train'

def mask_phase(path, leaf):
    return 'freeze' if 'phase' in path else 'train'

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
    
    return optimizer, trans_tree


def compare_params(old_params, new_params, atol=1e-12):
    def compare_fn(p_old, p_new):
        return not jax.numpy.allclose(p_old, p_new, atol=atol)

    diffs = jax.tree_util.tree_map(compare_fn, old_params, new_params)
    return diffs

def check_zero_grads(vstate, mask_fn=None, rtol=1e-12, atol=1e-14):
    """
    Comprueba si los gradientes de logψ en `vstate` son cero (u otra condición).
    
    Args:
        vstate: NetKet MCState
        mask_fn: función opcional que recibe el pytree de grads y devuelve 
                 un pytree booleano con True donde quieres chequear ceros.
                 Si None, se chequean todos.
        rtol, atol: tolerancias para `jnp.allclose`.
    
    Returns:
        report: diccionario con info sobre zeros/nans por parámetro.
    """
    params = vstate.parameters
    apply_fun = vstate._apply_fun
    s_batch = vstate.samples.reshape(-1, vstate.hilbert.size)
    
    def logpsi(p, s):
        return apply_fun({"params": p}, s)
    
    # Derivada de logψ wrt parámetros
    grad_logpsi = jax.grad(lambda p, s: jnp.real(logpsi(p, s)))
    
    # Evaluar en batch
    grads = jax.vmap(lambda s: grad_logpsi(params, s))(s_batch)
    
    # Reducir (media sobre muestras)
    grads_mean = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads)
    
    if mask_fn is None:
        mask_fn = lambda g: jax.tree_util.tree_map(lambda _: True, g)
    
    mask = mask_fn(grads_mean)
    
    def analyze(g, m):
        if not m: 
            return None
        return {
            "allclose_zero": jnp.allclose(g, 0.0, rtol=rtol, atol=atol),
            "has_nan": jnp.isnan(g).any(),
            "has_inf": jnp.isinf(g).any(),
            "max_abs": jnp.max(jnp.abs(g))
        }
    
    report = jax.tree_util.tree_map(analyze, grads_mean, mask)
    print(report)
    return 