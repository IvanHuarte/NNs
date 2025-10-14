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
            return {k: apply_mask(v, path + (k,)) for k, v in tree.items()}
        else:
            return predicate(path)

    return apply_mask(params)


def mask_modulus(path, leaf):
    return "freeze" if "modulus" in path else "train"


def mask_phase(path, leaf):
    return "freeze" if "phase" in path else "train"


def mask_both(path, leaf):
    return "freeze"


def masked_optimizer(params, transform_map, mode=None):
    """Genera una máscara universal para `params` basada en el modo.

    Args:
        params: Parámetros del modelo.
        mode: Parte a la que aplicar la máscara ('modulus','phase')
        transform_map: Mapa de transformaciones a aplicar.
    Returns:
        dict: Máscara con la misma estructura que `params`.
    """

    if mode == "modulus":
        trans_tree = flax.traverse_util.path_aware_map(mask_modulus, params)
    elif mode == "phase":
        trans_tree = flax.traverse_util.path_aware_map(mask_phase, params)
    elif mode == "both":
        trans_tree = flax.traverse_util.path_aware_map(mask_both, params)
    elif mode is None:
        return transform_map["train"]
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'modulus', 'phase' or None (for both)."
        )

    optimizer = optax.multi_transform(transform_map, trans_tree)

    return optimizer, trans_tree


def compare_params(old_params, new_params, atol=1e-13):
    def compare_fn(p_old, p_new):
        return not jax.numpy.allclose(p_old, p_new, atol=atol)

    diffs = jax.tree_util.tree_map(compare_fn, old_params, new_params)

    print(f"Ha cambiado: (True) //  No ha cambiado: (False) \n\n")
    print(diffs)


def summarize_report(report):
    from flax.traverse_util import flatten_dict

    flat = flatten_dict(report, sep="/")
    summary = {}
    all_zero = True
    for k, v in flat.items():
        if isinstance(v, dict) and "allclose_zero" in v:
            summary[k] = {
                "allclose_zero": bool(v["allclose_zero"]),
                "max_abs": float(v["max_abs"]),
            }
            if not bool(v["allclose_zero"]):
                all_zero = False
    return all_zero, summary


def check_zero_grads(vstate, branch, rtol=1e-12, atol=1e-14):
    """
    Comprueba si los gradientes de logψ en la rama 'modulus' o 'phase'
    del pytree de parámetros de `vstate` son cero.

    Args:
        vstate: NetKet MCState
        branch: 'modulus' o 'phase', la rama que debería estar congelada
        rtol, atol: tolerancias para jnp.allclose

    Returns:
        report: mismo pytree que params[branch], con info por tensor.
    """
    params = vstate.parameters
    apply_fun = vstate._apply_fun
    s_batch = vstate.samples.reshape(-1, vstate.hilbert.size)

    def logpsi(p, s):
        return apply_fun({"params": p}, s)

    grad_logpsi = jax.grad(lambda p, s: jnp.real(logpsi(p, s)))

    grads = jax.vmap(lambda s: grad_logpsi(params, s))(s_batch)
    grads_mean = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads)

    report = {}

    def analyze(g):
        return {
            "allclose_zero": jnp.allclose(g, 0.0, rtol=rtol, atol=atol),
            "has_nan": bool(jnp.isnan(g).any()),
            "has_inf": bool(jnp.isinf(g).any()),
            "max_abs": float(jnp.max(jnp.abs(g))),
        }

    def recurse(tree, path=()):
        if isinstance(tree, dict):
            for k, v in tree.items():
                if k == branch:
                    # Analizamos toda la subrama seleccionada
                    report["/".join(path + (k,))] = jax.tree_util.tree_map(analyze, v)
                else:
                    recurse(v, path + (k,))

    recurse(grads_mean)

    if not report:
        raise ValueError(f"No se encontró ninguna rama '{branch}' en los gradientes.")

    all_zero, summary = summarize_report(report)

    print("¿Toda la rama phase tiene gradientes ~0?", all_zero)
    for path, stats in summary.items():
        print(path, stats)

    return
