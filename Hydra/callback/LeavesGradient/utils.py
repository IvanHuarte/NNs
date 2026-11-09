import jax
import jax.numpy as jnp

criteria_dict_params = {
    "mean": lambda x: (
        "🟢"
        if (x > 1e-4 and x <= 1e1)
        else "🟡" if (x > 1e1 and x < 1e2) else "🔴" if x > 1e2 else "🟢"
    ),
    "std": lambda x: (
        "🟢"
        if (x > 1e-1 and x <= 1e1)
        else "🟡" if ((x > 1e1 and x < 3e1) or (x > 1e-2 and x < 1e-1)) else "🔴"
    ),
    "max": lambda x: (
        "🟢" if (x > 1e-1 and x <= 2e1) else "🟡" if (x > 2e1 and x < 1e2) else "🔴"
    ),
    "min": lambda x: (
        "🟢" if (x > 1e-5 and x <= 5e-2) else "🟡" if (x > 5e-2 and x < 1e0) else "🔴"
    ),
}

criteria_dict_gradients = {
    "mean": lambda x: (
        "🟢" if (x > 1e-4 and x <= 1e1) else "🟡" if (x > 1e1 and x < 1e2) else "🔴"
    ),
    "std": lambda x: (
        "🟢"
        if (x > 1e-1 and x <= 2e1)
        else "🟡" if ((x > 1e1 and x < 3e1) or (x > 1e-2 and x < 1e-1)) else "🔴"
    ),
    "max": lambda x: (
        "🟢" if (x > 1e-1 and x <= 3e1) else "🟡" if (x > 3e1 and x < 5e2) else "🔴"
    ),
    "min": lambda x: (
        "🟢" if (x > 1e-3 and x <= 2e1) else "🟡" if (x > 2e1 and x < 1e2) else "🔴"
    ),
}


def operation(name):
    if name == "mean":
        return jnp.mean
    elif name == "std":
        return jnp.std
    elif name == "max":
        return jnp.max
    elif name == "min":
        return jnp.min


def eval_criteria(criteria, gradients=True):
    func = (
        criteria_dict_gradients[criteria]
        if gradients
        else criteria_dict_params[criteria]
    )

    def nruter(leaf):
        return func(jnp.abs(leaf))

    return nruter


def display_collector(carry_leaf, leaf_value, leaf_status):
    return carry_leaf + f"  {leaf_value:.3e} {leaf_status}"


def display_header(metrics_list, show_parameters):
    metrics_str = " "
    for metric in metrics_list:
        metrics_str += f"    {metric}"

    len_metrics = len(metrics_str) // 2

    header = "\n"
    if show_parameters:
        header += (
            "**"
            + " " * (len_metrics - 5)
            + "PARAMETERS"
            + " " * (len_metrics - 5)
            + "    ||  "
            + " " * (len_metrics - 5)
            + "GRADIENTS"
            + " " * (len_metrics - 5)
            + "    **\n"
        )
        header += "**" + metrics_str + "    ||  " + metrics_str
    else:
        header += (
            "**"
            + " " * (len_metrics - 4)
            + "GRADIENTS"
            + " " * (len_metrics - 4)
            + "  **\n"
        )
        header += "**" + metrics_str

    header += "   **"

    return header


def append_tree_display(carry_tree, tree, metrics, gradients=True):

    for metric in metrics:
        metric_op = operation(metric)
        metric_tree_vals = jax.tree_util.tree_map(lambda x: metric_op(jnp.abs(x)), tree)
        metric_status = jax.tree_util.tree_map(
            eval_criteria(metric, gradients=gradients), metric_tree_vals
        )
        carry_tree = jax.tree_util.tree_map(
            display_collector, carry_tree, metric_tree_vals, metric_status
        )

    return carry_tree
