import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from netket.sampler.rules import MetropolisRule
from netket.utils.types import PyTree, PRNGKeyT
from netket.utils.struct import dataclass
from netket.hilbert.random import flip_state


@dataclass
class InvertMagnetization(MetropolisRule):
    """Monte Carlo mutation rule that inverts all the spins.

    Please refer to the NetKet API documentation for a detailed explanation of
    the MetropolisRule interface.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        indxs = jax.random.randint(
            key, shape=(1,), minval=0, maxval=sampler.n_chains
        )
        σp = σ.at[indxs, :].multiply(-1)
        return σp, None
    

class LocalRule_Z2(MetropolisRule):

    def random_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
        key: PRNGKeyT,
    ):
        """
        Generates a random state compatible with this rule.

        By default this calls :func:`netket.hilbert.random.random_state`.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            sampler_state: The current state of the sampler. Should not modify it.
            key: The PRNGKey to use to generate the random state.
        """
        raw_samples = sampler.hilbert.random_state(
            key, size=sampler.n_batches, dtype=sampler.dtype
        )
        raw_samples = raw_samples.at[:,:,0].set(1)
        print(raw_samples)
        return raw_samples
    
    def transition(rule, sampler, machine, parameters, state, key, σ):
        key1, key2 = jax.random.split(key, 2)

        n_chains = σ.shape[0]
        hilb = sampler.hilbert

        indxs = jax.random.randint(key1, shape=(n_chains,), minval=1, maxval=hilb.size)
        σp, _ = flip_state(hilb, key2, σ, indxs)
        σp = σp.at[:, 0].set(1)

        return σp, None