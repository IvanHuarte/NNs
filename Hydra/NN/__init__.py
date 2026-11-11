# MAIN WRAPPERS
from .ARNN import ARNNWrapper

############### NN.ARNN IMPORTATIONS #################
from .ARNN.SingleModules.ARViT import ARViT
from .NN import NNWrapper

############### NN.NN IMPORTATIONS #################
from .NN.FluxModules import (
    Sequential,
    Sequivariant,
    SequivariantX,
    SingleModule,
    SplitTraining,
    Transversal,
)

# Importation of single modules
from .NN.SingleModules import (
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
    PCvT,
    SiteHead,
    Sum,
    SzaboOutput,
    ViT2D,
    VViT,
)

# Importation of symmetrization modules
from .NN.Symm import Z2, Traslation
from .toolbox import CarreteSign, MarshallSign

############## NAMESPACES ################

__all_wrappers__ = [
    "NNWrapper",
    "ARNNWrapper",
]
__all_flux__ = [
    "SingleModule",
    "SplitTraining",
    "Sequential",
    "Sequivariant",
    "SequivariantX",
    "Transversal",
]

__all_single__ = [
    "ARViT",
    "CNN",
    "CNNLiang",
    "CvT",
    "CvT2",
    "CvT3",
    "CvTaps",
    "CvTexp",
    "PCvT",
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
    "SiteHead",
    "DeepOutputHead",
    "ComplexHead",
    "SzaboOutput",
    "Sum",
    "Mean",
    "FinalFF",
]

__all_symm__ = [
    "Z2",
    "Traslation",
    "TraslationAnchor",
]

__all_AR__ = ["ARViT"]

__all__ = __all_flux__ + __all_single__ + __all_symm__ + __all_AR__

# Diccionarios de registro
REGISTRY_WRAPPERS = {
    "NNWrapper": NNWrapper,
    "ARNNWrapper": ARNNWrapper,
}

REGISTRY_FLUX = {
    "SingleModule": SingleModule,
    "SplitTraining": SplitTraining,
    "Sequential": Sequential,
    "Sequivariant": Sequivariant,
    "SequivariantX": SequivariantX,
    "Transversal": Transversal,
}

REGISTRY_SINGLE = {
    "ARViT": ARViT,
    "CNN": CNN,
    "CNNLiang": CNNLiang,
    "CvT": CvT,
    "CvT2": CvT2,
    "CvT3": CvT3,
    "CvTaps": CvTaps,
    "CvTexp": CvTexp,
    "PCvT": PCvT,
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
    "SiteHead": SiteHead,
    "DeepOutputHead": DeepOutputHead,
    "ComplexHead": ComplexHead,
    "FinalFF": FinalFF,
    "SzaboOutput": SzaboOutput,
    "Sum": Sum,
    "Mean": Mean,
}
REGISTRY_SYMM = {
    "Z2": Z2,
    "Traslation": Traslation,
}


REGISTRY = {
    None: None,
    **REGISTRY_SINGLE,
    **REGISTRY_FLUX,
    **REGISTRY_SYMM,
    **REGISTRY_WRAPPERS,
}

# Diccionarios de modulos que necesitan argumentos externos
LATTICE_SIZE = {
    "lattice_size": [
        "CNN",
        "CNNLiang",
        "CvTaps",
        "CvTexp",
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

HILBERT = {"hilbert": ["ARViT"]}

# Modulos de simetrizacion
LATTICE_SIZE["lattice_size"].extend(
    (
        "ARNNWrapper",
        "NNWrapper",
        "Traslation",
    )
)
HILBERT["hilbert"].extend(("ARNNWrapper",))


EXTERNAL_ARGS = {**LATTICE_SIZE, **HILBERT}

# Tags and relations used in Neural Network initialization
wrapper_submodule_dict = {
    "ARNNWrapper": "ARNN",
    "NNWrapper": "NN",
}

flux_submodule_dict = {
    "SingleModule": "Single",
    "SplitTraining": ["Modulus", "Phase"],
    "Sequential": "Seq",
    "Sequivariant": "SeqV",
    "SequivariantX": "SeqVX",
    "Transversal": "Trans",
}
symm_submodule_dict = {
    "Z2": "Z2Wrap",
    "Traslation": "TWrap",
}

undefined_submodule_number = [
    "Sequential",
    "Sequivariant",
    "SequivariantX",
    "Transversal",
]

submodule_tags = [
    "Wrap",  # symmetry wrappers (Z2, Traslation) built from symm_wrapper
    "ARNN",
    "NN",
    "Single",
    "Modulus",
    "Phase",
    "Seq",
    "SeqV",
    "SeqVX",
    "Trans",
]


def get_submodules(father, n_mod):
    if father == "NNWrapper":
        submodules = ["NN"]
    if father == "ARNNWrapper":
        submodules = ["ARNN"]
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
