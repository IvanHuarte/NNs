import jax


def print_tree(tree, prefix="", values=False):
    for key, val in tree.items():

        if isinstance(val, dict):
            print(prefix + str(key))
            print_tree(val, prefix + "   ", values=values)
        else:
            if values:
                print(prefix + f"{str(key)}: {str(val)}")
            else:
                print(prefix + str(key))


def compare_params(old_params, new_params, atol=1e-13):
    def compare_fn(p_old, p_new):
        return not jax.numpy.allclose(p_old, p_new, atol=atol)

    diffs = jax.tree_util.tree_map(compare_fn, old_params, new_params)

    print(f"Ha cambiado: (True) //  No ha cambiado: (False) \n\n")
    print(diffs)
    print("\n")
