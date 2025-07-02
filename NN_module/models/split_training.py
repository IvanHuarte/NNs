from typing import Callable, Sequence, Tuple, Any

import flax.linen as nn
import jax.numpy as jnp

from .MLP import BatchedMultiLayerPerceptron
from .ViT_2D import BatchedSpinViT

DTYPE = jnp.float64

class SplitTraining_ViT_MLP(nn.Module):

    """
    Flax module to train module and phase separately
    """

    lattice_size: Tuple[int, int]

    "Module settings ViT"
    token_size: Tuple[int, int]
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool = False
    symm_2D_module: bool = False
    symm_Z2_module: bool = False
    trivial_Z2_module: bool = True

    "Phase settings MLP"
    param_dtype_phase : Any = DTYPE
    hidden_alpha: int | Tuple[int, ...] = None
    activation: Callable | Tuple[Callable, ...] = None
    output_dim: int = 1
    symm_2D_phase: bool = False
    symm_Z2_phase: bool = False
    trivial_Z2_phase: bool = True

    train_phase: bool = False

    @nn.compact
    def __call__(self, batch_x, a=None):

        print(a)

        # Module Training
        log_module = BatchedSpinViT(
            lattice_size=self.lattice_size,
            token_size=self.token_size,
            embedding_d=self.embedding_d,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            n_ffn_layers=self.n_ffn_layers,
            final_architecture=self.final_architecture,
            is_complex=self.is_complex,
            symm_2D = self.symm_2D_module,
            symm_Z2 = self.symm_Z2_module,
            trivial_Z2 = self.trivial_Z2_module
        )(batch_x)


        # Phase Training
        if self.train_phase:
            phase = BatchedMultiLayerPerceptron(
                lattice_size=self.lattice_size,
                param_dtype=self.param_dtype_phase,
                hidden_alpha=self.hidden_alpha,
                activation=self.activation,
                symm_2D=self.symm_2D_phase,
                symm_Z2=self.symm_Z2_phase,
                trivial_Z2=self.trivial_Z2_phase
            )(batch_x)

            return log_module + 1j * phase
        
        return log_module
