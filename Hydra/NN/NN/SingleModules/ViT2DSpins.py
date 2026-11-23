"""
ViT2DSpins: ViT2D whose feature grid also carries the spins of each token.

Output (B, H, W, C + sy*sx): the ViT2D feature grid (final_architecture=None)
followed by the sy*sx spins (+-1) of each token, in the Tokenize order
k = a*sx + b for the spin at (i*sy + a, j*sx + b). It lets a head use the
configuration itself, e.g. SpinMPSHead(physical=True) contracts its MPS
tensors with the spin (the physical index of Fan et al., arXiv:2603.14425)
and SpinMPSHead(marshall=True) multiplies by the Marshall sign.

Same parameters as ViT2D (a subclass with no new ones), so ViT2D states load
into it. Channel 0 (the momentum factor z of Sequivariant) is the ViT2D one.
Meant for those SpinMPSHead options: an OutputHead on top would also pool the
spin channels.
"""

import jax.numpy as jnp

from .ViT2D import Tokenize, ViT2D


class ViT2DSpins(ViT2D):
    """ViT2D (same fields) with the token spins appended as the last sy*sx channels."""

    def __call__(self, x):
        if self.final_architecture is not None:
            raise ValueError("ViT2DSpins needs final_architecture=None (a feature grid)")
        f = super().__call__(x)
        sy, sx = self.token_size
        spins = Tokenize(self.token_size, f.shape[1:3], sy * sx)(x)
        return jnp.concatenate([f, spins.astype(f.dtype)], axis=-1)
