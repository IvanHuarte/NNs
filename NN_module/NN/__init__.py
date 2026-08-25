# Importation of factories
from .Factories import (
    Sequential,
    Sequivariant,
    SequivariantX,
    SingleModule,
    SplitTraining,
    Transversal,
)

# Importation of single modules
from .SingleModels import (
    CMLP,
    CNN,
    MLP,
    CNNbinClsf,
    CNNClsf,
    CNNLiang,
    CNNPh,
    CNNSzabo,
    ComplexHead,
    CvT,
    CvT2,
    CvT3,
    CvTaps,
    CvTexp,
    DeepOutputHead,
    EDPPh,
    Factorized,
    FactorMod,
    FinalFF,
    Jastrow_wrap,
    Mean,
    OutputHead,
    Sum,
    SzaboOutput,
    ViT2D,
    VViT,
)

# Importation of symmetrization modules
from .Symm import Z2, SymmWrapper, Traslation
from .toolbox import CarreteSign, MarshallSign

# Final Architecture modules


# Listas de exportación
__all_factories__ = [
    "SingleModule",
    "SplitTraining",
    "Sequential",
    "Sequivariant",
    "SequivariantX",
    "Transversal",
]

__all_single__ = [
    "CNN",
    "CNNLiang",
    "CvT",
    "CvT2",
    "CvT3",
    "CvTaps",
    "CvTexp",
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
    "VViT",
    "Factorized",
    "FactorMod",
    "Jastrow_wrap",
    "OutputHead",
    "DeepOutputHead",
    "ComplexHead",
    "SzaboOutput",
    "Sum",
    "Mean",
    "FinalFF",
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
    "Sequivariant": Sequivariant,
    "SequivariantX": SequivariantX,
    "Transversal": Transversal,
}

REGISTRY_SINGLE = {
    "CNN": CNN,
    "CNNLiang": CNNLiang,
    "CvT": CvT,
    "CvT2": CvT2,
    "CvT3": CvT3,
    "CvTaps": CvTaps,
    "CvTexp": CvTexp,
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
    "VViT": VViT,
    "Factorized": Factorized,
    "FactorMod": FactorMod,
    "Jastrow_wrap": Jastrow_wrap,
    "OutputHead": OutputHead,
    "DeepOutputHead": DeepOutputHead,
    "ComplexHead": ComplexHead,
    "FinalFF": FinalFF,
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
        "CNNPh",
        "EDPPh",
        "CNNClsf",
        "CNNbinClsf",
        "CNNSzabo",
        "MarshallSign",
        "CarreteSign",
        "Factorized",
        "FactorMod",
        "SzaboOutput",
    ]
}

# Modulos de simetrizacion
LATTICE_SIZE["lattice_size"].extend(
    (
        "SymmWrapper",
        "Traslation",
    )
)

EXTERNAL_ARGS = {**LATTICE_SIZE}

# Tags and relations used in Neural Network initialization
factory_submodule_dict = {
    "SingleModule": "Single",
    "SplitTraining": ["Modulus", "Phase"],
    "Sequential": "Seq",
    "Sequivariant": "SeqV",
    "SequivariantX": "SeqVX",
    "Transversal": "Trans",
}
symm_submodule_dict = {
    "SymmWrapper": "SymmModel",
    "Z2": "Z2Wrap",
    "Traslation": "TWrap",
}

undefined_submodule_number = [
    "Sequential",
    "Sequivariant",
    "SequivariantX",
    "Transversal",
]

factory_submodule_tags = [
    "Wrap",
    "Single",
    "Modulus",
    "Phase",
    "Seq",
    "SeqV",
    "SeqVX",
    "Trans",
]


def get_submodules(father, n_mod):
    if father == "Sequential":
        submodules = [f"Seq_{i}" for i in range(n_mod)]
    if father == "Sequivariant":
        submodules = [f"SeqV_{i}" for i in range(n_mod)]
    if father == "SequivariantX":
        submodules = [f"SeqVX_{i}" for i in range(n_mod)]
    elif father == "Transversal":
        submodules = [f"Trans_{i}" for i in range(n_mod)]
    elif father == "SplitTraining":
        submodules = ["Modulus", "Phase"]
    elif father == "SingleModule":
        submodules = ["Single"]
    return submodules
