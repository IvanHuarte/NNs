# Importation of factories
from .Factories.SingleModule import SingleModule
from .Factories.SplitTraining import SplitTraining
from .Factories.Sequential import Sequential
from .Factories.Transversal import Transversal

# Importation of single modules
from .SingleModels.CNN import CNN
from .SingleModels.CNNLiang import CNNLiang
from .SingleModels.CvT import CvT
from .SingleModels.CvT2 import CvT2
from .SingleModels.CvTaps import CvTaps
from .SingleModels.MLP import MLP
from .SingleModels.CMLP import CMLP
from .SingleModels.Phase import CNNPh, EDPPh, CNNClsf, CNNbinClsf, CNNSzabo
from .SingleModels.ViT2D import ViT2D
from .SingleModels.ViT import ViT
from .SingleModels.VViT import VViT

from .SingleModels.Ansatz import Factorized, FactorMod, Jastrow_wrap
from .SingleModels.FluxFunc import Sum, Mean, OutputHead, SzaboOutput

from .toolbox import MarshallSign, CarreteSign

# Importation of symmetrization modules
from .Symm import SymmWrapper
from .Symm import Z2, Traslation

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
    "CvT2",
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
    "VViT" "Factorized",
    "FactorMod",
    "Jastrow_wrap",
    "OutputHead",
    "SzaboOutput",
    "Sum",
    "Mean",
]

__all_symm__ = [
    "SymmWrapper",
    "Z2",
    "Traslation",
    "TraslationAnchor",
]

__all__ = __all_factories__ + __all_single__ + __all_symm__

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
    "CvT2": CvT2,
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
    "OutputHead": OutputHead,
    "SzaboOutput": SzaboOutput,
    "Sum": Sum,
    "Mean": Mean,
}
REGISTRY_SYMM = {
    "SymmWrapper": SymmWrapper,
    "Z2": Z2,
    "Traslation": Traslation,
}

REGISTRY = {None: None, **REGISTRY_SINGLE, **REGISTRY_FACTORIES, **REGISTRY_SYMM}


# Diccionarios de modulos que necesitan argumentos externos
LATTICE_SIZE = {
    "lattice_size": [
        "CNN",
        "CNNLiang",
        "CvT",
        "CvT2",
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
        "SzaboOutput",
    ]
}

# Modulos de simetrizacion
LATTICE_SIZE["lattice_size"].append(["Traslation", "TraslationAnchor"])

EXTERNAL_ARGS = {**LATTICE_SIZE}


# Tags used in configurations
factory_submodule_dict = {
    "SingleModule": "single",
    "SplitTraining": ["modulus", "phase"],
    "Sequential": "Seq",
    "Transversal": "Trans",
}

factory_submodule_tags = ["Single", "ModulusNet", "PhaseNet", "Seq", "Trans"]
