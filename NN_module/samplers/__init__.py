from NN_module.samplers.custom_rules import LocalRule_Z2, InvertMagnetization

from netket.sampler.rules import LocalRule

__all__ = ['LocalRule','LocalRule_Z2', 'InvertMagnetization']

RULES = {
    'LocalRule': LocalRule,
    'LocalRule_Z2': LocalRule_Z2,
    'InvertMagnetization': InvertMagnetization,
}