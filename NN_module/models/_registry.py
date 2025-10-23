# Module for automatically incorporate new (and old) flax modules via decorators

NN_REGISTRY = {"None": None}


def register_module(name):
    def decorator(cls):
        NN_REGISTRY[name] = cls
        return cls

    return decorator
