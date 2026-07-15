import flax.linen as nn
import jax.numpy as jnp


class SplitTraining(nn.Module):
    """
    Flax module to train modulus and phase separately
    """

    Modulus: nn.Module
    Phase: nn.Module

    def setup(self):
        self.Modulus_model = self.Modulus
        self.Phase_model = self.Phase

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        # print(f"Modulus")
        log_modulus = self.Modulus_model(x)

        # print(f"Phase")
        phase = self.Phase_model(x)

        # print(f"Modulus: {log_modulus.shape}")
        # print(f"Phase: {phase.shape}")
        # print(self.squeeze)

        return log_modulus + 1.0j * phase
