from NN_module.sampler.custom_rules import (
    LocalRule_Z2,
    InvertMagnetization,
    ExchangeJ1J2,
    Exchange,
)

from netket.sampler.rules import LocalRule

__all__ = [
    "LocalRule",
    "LocalRule_Z2",
    "InvertMagnetization",
    "ExchangeJ1J2",
    "Exchange",
]

RULES = {
    "LocalRule": LocalRule,
    "LocalRule_Z2": LocalRule_Z2,
    "InvertMagnetization": InvertMagnetization,
    "ExchangeJ1J2": ExchangeJ1J2,
    "Exchange": Exchange,
}
