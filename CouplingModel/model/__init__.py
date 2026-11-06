from .model import IsingSquare, J1J2Square, LRChain, LRSquare, Oxalate, SSMSquare

__all__ = ["Oxalate", "LRChain", "LRSquare", "SSMSquare", "J1J2Square", "IsingSquare"]

REGISTRED_MODELS = {
    "Oxalate": Oxalate,
    "LRChain": LRChain,
    "LRSquare": LRSquare,
    "SSMSquare": SSMSquare,
    "J1J2Square": J1J2Square,
    "IsingSquare": IsingSquare,
}
