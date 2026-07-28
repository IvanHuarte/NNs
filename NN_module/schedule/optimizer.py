import optax

optimizer_dict = {
    "sgd": optax.sgd,
    "adam": optax.adam,
}

leaf_gradient_modificators = {
    "clip": optax.clip,
    "clip_by_block_rms": optax.clip_by_block_rms,
    "scale": optax.scale,
    "scale_by_rms": optax.scale_by_rms,
    "scale_by_learning_rate": optax.scale_by_learning_rate,
    "add_noise": optax.add_noise,
    "trace": optax.trace,
}


def build_leaf_optimizer(optimizer_setup, lr_function):

    opt_method = optimizer_setup["optimizer"]
    modifications = optimizer_setup["modifications"]

    optimizer_method = optimizer_dict[opt_method]
    considered_modifications = dict(
        [(k, v) for k, v in modifications.items() if v not in [None, False]]
    )

    modifications_in_chain = []
    for mod, value in considered_modifications.items():
        mood_function = leaf_gradient_modificators[mod]

        if isinstance(value, list):
            modifications_in_chain.append(mood_function(*value))

        else:
            modifications_in_chain.append(mood_function(value))

    return optax.chain(optimizer_method(lr_function), *modifications_in_chain)


def get_transformed_optimizer(optimizer_setup, modes, lr_func):

    transformation = {"freeze": optax.set_to_zero()}

    for branch, lr_f in zip(modes, lr_func):
        transformation[f"train_{branch}"] = build_leaf_optimizer(optimizer_setup, lr_f)

    return transformation
