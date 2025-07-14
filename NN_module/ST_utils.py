
class WrappedModel:
    def __init__(self, model, train_modulus, train_phase):
        self.model = model
        self.train_modulus = train_modulus
        self.train_phase = train_phase

    def init(self, rng, x):
        return self.model.init(rng, x,
                               train_modulus=self.train_modulus,
                               train_phase=self.train_phase)

    def apply(self, vars, x, **kwargs):
        return self.model.apply(vars, x,
                                train_modulus=self.train_modulus,
                                train_phase=self.train_phase,
                                **kwargs)

def init_function(model, modulus=True, phase=True):
    
    def core_init(rng, dummy_x, **kwargs):
        return model.init(rng, dummy_x, train_modulus=modulus, train_phase=phase, **kwargs)
    return core_init

def apply_function(model, modulus=True, phase=True):

    def core_apply(variables, x, **kwargs):
        return model.apply(variables, x, train_modulus=modulus, train_phase=phase, **kwargs)
    return core_apply


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

