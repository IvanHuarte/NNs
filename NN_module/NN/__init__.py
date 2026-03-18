# Importation of factories
from .Factories import (
    SingleModule,
    SplitTraining,
    Sequential,
    Transversal,
)

# Importation of single modules
from .SingleModels import (
    CNN,
    CNNLiang,
    CvT,
    CvT2,
    CvTaps,
    MLP,
    CMLP,
    ViT2D,
    ViT,
    VViT,
    CNNPh,
    EDPPh,
    CNNClsf,
    CNNbinClsf,
    CNNSzabo,
    Factorized,
    FactorMod,
    Jastrow_wrap,
    Sum,
    Mean,
    OutputHead,
    ComplexHead,
    SzaboOutput,
)

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
    "VViT",
    "Factorized",
    "FactorMod",
    "Jastrow_wrap",
    "OutputHead",
    "ComplexHead",
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
    "ComplexHead": ComplexHead,
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
        "ViT2D",
        "ViT",
        "Factorized",
        "FactorMod",
        "SzaboOutput",
    ]
}

# Modulos de simetrizacion
LATTICE_SIZE["lattice_size"].append(*("Traslation",))

EXTERNAL_ARGS = {**LATTICE_SIZE}

# Tags and relations used in Neural Network initialization
factory_submodule_dict = {
    "SingleModule": "Single",
    "SplitTraining": ["Modulus", "Phase"],
    "Sequential": "Seq",
    "Transversal": "Trans",
}
symm_submodule_dict = {
    "SymmWrapper": "SymmModel",
    "Z2": "Z2Wrap",
    "Traslation": "TWrap",
}

undefined_submodule_number = ["Sequential", "Transversal"]

factory_submodule_tags = ["Wrap", "Single", "Modulus", "Phase", "Seq", "Trans"]
