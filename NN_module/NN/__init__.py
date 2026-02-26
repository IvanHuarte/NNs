# Importation of factories
from .Factories.SingleModule import SingleModule
from .Factories.SplitTraining import SplitTraining
from .Factories.Sequential import Sequential
from .Factories.Transversal import Transversal

# Importation of single modules
from .SingleModels.CNN import CNN
from .SingleModels.CNNLiang import CNNLiang
from .SingleModels.CvT import CvT
from .SingleModels.CvTaps import CvTaps
from .SingleModels.MLP import MLP
from .SingleModels.CMLP import CMLP
from .SingleModels.Phase import CNNPh, EDPPh, CNNClsf, CNNbinClsf, CNNSzabo
from .SingleModels.ViT2D import ViT2D
from .SingleModels.ViT import ViT
from .SingleModels.VViT import VViT

from .SingleModels.Ansatz import Factorized, FactorMod, Jastrow_wrap
from .SingleModels.FluxFunc import Sum, Mean

from .toolbox import MarshallSign, CarreteSign

# Final Architecture modules


# Listas de exportación
__all_factories__ = [
    "SingleModule",
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
    "CMLP",
    "ViT2D",
    "ViT",
    "VViT"
    "Factorized",
    "FactorMod",
    "Jastrow_wrap",
    "Sum",
    "Mean",
]

__all__ = __all_factories__ + __all_single__

# Diccionarios de registro
REGISTRY_FACTORIES = {
    "SingleModule": SingleModule,
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
    "CMLP": CMLP,
    "ViT2D": ViT2D,
    "ViT": ViT,
    "VViT": VViT,
    "Factorized": Factorized,
    "FactorMod": FactorMod,
    "Jastrow_wrap": Jastrow_wrap,
    "Sum": Sum,
    "Mean": Mean,
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
        "CMLP",
        "ViT2D",
        "ViT",
        "Factorized",
        "FactorMod",
    ]
}

EXTERNAL_ARGS = {**LATTICE_SIZE}


# Tags used in configurations
factory_submodule_dict = {
    "SingleModule": "single",
    "SplitTraining": ["modulus", "phase"],
    "Sequential": "Seq",
    "Transversal": "Trans",
}

factory_submodule_tags = ["Single", "ModulusNet", "PhaseNet", "Seq", "ZZ", "Trans"]
