import jax
import flax


def is_subsequence(a, b):
    n, m = len(a), len(b)
    for i in range(m - n + 1):
        if b[i : i + n] == a:
            return True
    return False


def train_branches(mode_paths, path2code):
    def _callable(path, leaf):
        m = next(
            (
                f"train_{path2code[mod_path]}"
                for mod_path in mode_paths
                if is_subsequence(mod_path, path)
            ),
            "freeze",
        )
        return m

    return _callable


def init_mask(path, leaf):
    return "-1"


def merge_trees(leaf_1, leaf_2):
    return leaf_1 if leaf_1 is not None else leaf_2 if leaf_2 is not None else None


def masked_optimizer(params, mode_paths, path2code):
    """Genera una máscara universal para `params` basada en el modo.

    Args:
        params: Parámetros del modelo.
        mode: Parte a la que aplicar la máscara ('modulus'/'phase'/'both'/None)
        transform_map: Mapa de transformaciones a aplicar.
    Returns:
        Pytree: Máscara con la misma estructura que `params`.

    """

    # init_tree = flax.traverse_util.path_aware_map(init_mask, params)

    train_tree = flax.traverse_util.path_aware_map(
        train_branches(mode_paths, path2code), params
    )

    # jax.tree_util.tree_map(merge_trees, init_tree, train_tree)
    # TODO
    # Explict masking of a branch
    # trans_tree = flax.traverse_util.path_aware_map(
    #     mask_branch(mask_paths, path2code), train_tree
    # )

    assert jax.tree_util.tree_reduce(
        lambda a, b: a & b,
        jax.tree_util.tree_map(lambda leaf: leaf != "-1", train_tree),
    )
    return train_tree
