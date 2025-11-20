# Importation of factories
from .Factories.SplitTraining import SplitTraining
from .Factories.Sequential import Sequential
from .Factories.Transversal import Transversal

# Importation of single modules
from .SingleModels.CNN import CNN
from .SingleModels.CNNLiang import CNNLiang
from .SingleModels.CvT import CvT
from .SingleModels.CvTaps import CvTaps
from .SingleModels.MLP import MLP
from .SingleModels.Phase import CNNPh, EDPPh, CNNClsf, CNNbinClsf
from .SingleModels.ViT2D import ViT2D
from .SingleModels.ViT import ViT

from .toolbox import MarshallSign

# Final Architecture modules


# Listas de exportación
__all_factories__ = [
    "SplitTraining",
    "Sequential",
    "Transversal",
]

__all_single__ = [
    "CNN",
    "CNNLiang",
    "CvT",
    "CvTaps",
    "CNNPh",
    "EDPPh",
    "CNNClsf",
    "CNNbinClsf",
    "MarshallSign",
    "MLP",
    "ViT2D",
    "ViT",
]

__all__ = __all_factories__ + __all_single__

# Diccionarios de registro
REGISTRY_FACTORIES = {
    "SplitTraining": SplitTraining,
    "Sequential": Sequential,
    "Transversal": Transversal,
}

REGISTRY_SINGLE = {
    "CNN": CNN,
    "CNNLiang": CNNLiang,
    "CvT": CvT,
    "CvTaps": CvTaps,
    "CNNPh": CNNPh,
    "EDPPh": EDPPh,
    "CNNClsf": CNNClsf,
    "CNNbinClsf": CNNbinClsf,
    "MarshallSign": MarshallSign,
    "MLP": MLP,
    "ViT2D": ViT2D,
    "ViT": ViT,
}

REGISTRY = {None: None, **REGISTRY_SINGLE, **REGISTRY_FACTORIES}


# Diccionarios de modulos que necesitan argumentos externos

LATTICE_SIZE = {
    "lattice_size": [
        "CNN",
        "CNNLiang",
        "CvT",
        "CvTaps",
        "CNNPh",
        "EDPPh",
        "CNNClsf",
        "CNNbinClsf",
        "MarshallSign",
        "MLP",
        "ViT2D",
        "ViT",
    ]
}

EXTERNAL_ARGS = {**LATTICE_SIZE}
