# Importation of single modules
from .CNN import CNN
from .CvT import CvT
from .CvT2 import CvT2
from .CvT3 import CvT3
from .CvTaps import CvTaps
from .MLP import MLP
from .Phase import CNNPh, EDPPh, CNNClsf
from .ViT_2D import ViT2D
from .ViT import ViT

# Importation of factories
from .Factories.SplitTraining import SplitTraining
from .Factories.Sequential import Sequential


__all__ = [
    "CNN",
    "CvT",
    "CvT2",
    "CvT3",
    "CvTaps",
    "CNNPh",
    "EDPPh",
    "CNNClsf",
    "MLP",
    "ViT2D",
    "ViT",

    "SplitTraining",
    "Sequential"
]

REGISTRY = {
    None : None,
    "CNN":CNN,
    "CvT":CvT,
    "CvT2":CvT2,
    "CvT3":CvT3,
    "CvTaps":CvTaps,
    "CNNPh":CNNPh,
    "EDPPh":EDPPh,
    "CNNClsf":CNNClsf,
    "MLP":MLP,
    "ViT2D":ViT2D,
    "ViT":ViT,
    
    "SplitTraining":SplitTraining,
    "Sequential":Sequential,
}