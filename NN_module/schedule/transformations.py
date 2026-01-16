import optax


def SplitTraining_transform(params, optimizer, lr_func):

    if len(lr_func) < 2:
        transformation = {
            "freeze": optax.set_to_zero(),
            "train": optimizer(lr_func[0]),
            "train_modulus": optimizer(lr_func[0]),
            "train_phase": optimizer(lr_func[0]),
        }
    else:
        transformation = {
            "freeze": optax.set_to_zero(),
            "train": optimizer(lr_func[0]),
            "train_modulus": optimizer(lr_func[0]),
            "train_phase": optimizer(lr_func[1]),
        }

    return transformation

def Sequential_transformation(params, optimizer, lr_func):

    transformation = {
        "freeze": optax.set_to_zero()
    }

    for branch, lr_f in zip([])



    "train": optimizer(lr_func[0]),
    "train_modulus": optimizer(lr_func[0]),
    "train_phase": optimizer(lr_func[1]),