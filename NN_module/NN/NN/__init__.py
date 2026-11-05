import flax.linen as nn
import jax.numpy as jnp
import netket as nk

from .FluxModules import (
    Sequential,
    Sequivariant,
    SequivariantX,
    SingleModule,
    SplitTraining,
    Transversal,
)
from .SingleModules import (
    CMLP,
    CNN,
    MLP,
    CNNLiang,
    ComplexHead,
    CvT,
    CvT2,
    CvT3,
    CvTaps,
    CvTexp,
    DeepOutputHead,
    FinalFF,
    Mean,
    OutputHead,
    Sum,
    SzaboOutput,
    ViT2D,
    VViT,
)
from .Symm import Z2, Traslation


# Wrapper over an Autoregressive Neural Network
class NNWrapper(nn.Module):
    """
    A wrapper for symmetry-based modules. It takes care of the shape
    of the input and ensures that the shape (B, N) is reshaped into
    (B, Lx, Ly, 1), which is expected by the deeper modules.

    Also ensures the (B,) shape for the output.
    """

    NN: nn.Module
    lattice_size: tuple[int, int]

    def setup(self):
        self.NN_model = self.NN

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        x = x.reshape(x.shape[0], *self.lattice_size, 1)

        x = self.NN_model(x)

        return jnp.atleast_1d(jnp.squeeze(x))
