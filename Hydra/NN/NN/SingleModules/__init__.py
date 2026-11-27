# Importation of single modules
from .Ansatz import Factorized, FactorMod, Jastrow_wrap
from .CMLP import CMLP
from .CNN import CNN
from .CNNLiang import CNNLiang
from .CvT import CvT
from .CvT2 import CvT2
from .CvT3 import CvT3
from .CvTsquad import CvTsquad
from .CvTaps import CvTaps
from .CvTexp import CvTexp
from .FluxFunc import (
    ComplexHead,
    DeepOutputHead,
    FinalFF,
    Mean,
    OutputHead,
    Sum,
    SzaboOutput,
)
from .MLP import MLP
from .MPSHead import MPSHead
from .PCvT import PCvT, SiteHead
from .PEPSHead import PEPSHead
from .SpinMPSHead import SpinMPSHead
from .SpinTTNHead import SpinTTNHead
from .SpinBTTNHead import SpinBTTNHead
from .Phase import CNNbinClsf, CNNClsf, CNNPh, CNNSzabo, EDPPh
from .ViT2D import ViT2D
from .ViT2DSpins import ViT2DSpins
from .VViT import VViT
