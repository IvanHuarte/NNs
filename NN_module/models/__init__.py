# Importation of factories
from .Factories.SplitTraining import SplitTraining
from .Factories.Sequential import Sequential
from .Factories.Transversal import Transversal

# Importation of single modules
from .SingleModels.CNN import CNN
from .SingleModels.CvT import CvT
from .SingleModels.CvTaps import CvTaps
from .SingleModels.MLP import MLP
from .SingleModels.Phase import CNNPh, EDPPh, CNNClsf, CNNbinClsf
from .SingleModels.ViT_2D import ViT2D
from .SingleModels.ViT import ViT

# Final Architecture modules


__all__ = [
    "CNN",
    "CvT",
    "CvTaps",
    "CNNPh",
    "EDPPh",
    "CNNClsf",
    "CNNbinClsf",
    "MLP",
    "ViT2D",
    "ViT",
    "SplitTraining",
    "Sequential",
    "Transversal",
]

REGISTRY = {
    None: None,
    "CNN": CNN,
    "CvT": CvT,
    "CvTaps": CvTaps,
    "CNNPh": CNNPh,
    "EDPPh": EDPPh,
    "CNNClsf": CNNClsf,
    "CNNbinClsf": CNNbinClsf,
    "MLP": MLP,
    "ViT2D": ViT2D,
    "ViT": ViT,
    "SplitTraining": SplitTraining,
    "Sequential": Sequential,
    "Transversal": Transversal,
}
