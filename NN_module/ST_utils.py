import jax.numpy as jnp

def init_function(model, modulus=True, phase=True):
    
    def core_init(rng, dummy_x, **kwargs):
        return model.init(rng, dummy_x, train_modulus=modulus, train_phase=phase, **kwargs)
    return core_init

def apply_function(model, modulus=True, phase=True):

    def core_apply(variables, x, **kwargs):
        return model.apply(variables, x, train_modulus=modulus, train_phase=phase, **kwargs)
    return core_apply

