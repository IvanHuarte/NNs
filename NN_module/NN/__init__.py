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
from .SingleModels.Phase import CNNPh, EDPPh, CNNClsf, CNNbinClsf, CNNSzabo
from .SingleModels.ViT2D import ViT2D
from .SingleModels.ViT import ViT
from .SingleModels.Ansatz import Factorized, FactorMod, Jastrow_wrap

from .toolbox import MarshallSign, CarreteSign

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
    "CNNSzabo",
    "MarshallSign",
    "CarreteSign",
    "MLP",
    "ViT2D",
    "ViT",
    "Factorized",
    "FactorMod",
    "Jastrow_wrap",
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
    "CNNSzabo": CNNSzabo,
    "MarshallSign": MarshallSign,
    "CarreteSign": CarreteSign,
    "MLP": MLP,
    "ViT2D": ViT2D,
    "ViT": ViT,
    "Factorized": Factorized,
    "FactorMod": FactorMod,
    "Jastrow_wrap": Jastrow_wrap,
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
        "CNNSzabo",
        "MarshallSign",
        "CarreteSign",
        "MLP",
        "ViT2D",
        "ViT",
        "Factorized",
        "FactorMod"
    ]
}

EXTERNAL_ARGS = {**LATTICE_SIZE}


# Tags used in configurations
factory_submodule_dict = {
    "SplitTraining": ["modulus", "phase"],
    "Sequential": "Seq",
    "Transversal": "Trans",
}

factory_submodule_tags = ["ModulusNet", "PhaseNet", "Seq", "ZZ", "Trans"]
