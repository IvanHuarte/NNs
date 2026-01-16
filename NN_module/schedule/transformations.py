import optax

def transformation_dictionary(optimizer, modes, lr_func):

    transformation = {
        "freeze": optax.set_to_zero()
    }

    for branch, lr_f in zip(modes, lr_func):
        transformation[branch] = optimizer(lr_f)

    return transformation

