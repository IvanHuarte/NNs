import jax
import jax.numpy as jnp
from flax import linen as nn
import netket
from netket.sampler.rules import MetropolisRule
from netket.utils.types import PyTree, PRNGKeyT
from netket.utils.struct import dataclass
from netket.hilbert.random import flip_state
from netket.utils import struct
from typing import override

from .initial_states import hilbert_state, M0_state, Mmax_state


@dataclass
class InvertMagnetization(MetropolisRule):
    """Monte Carlo mutation rule that inverts all the spins.

    Please refer to the NetKet API documentation for a detailed explanation of
    the MetropolisRule interface.
    """

    @override
    def transition(rule, sampler, machine, parameters, state, key, σ):
        indxs = jax.random.randint(key, shape=(1,), minval=0, maxval=sampler.n_chains)
        σp = σ.at[indxs, :].multiply(-1)
        return σp, None


class LocalRule_Z2(MetropolisRule):

    @override
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
        raw_samples = raw_samples.at[:, :, 0].set(1)
        return raw_samples

    def transition(rule, sampler, machine, parameters, state, key, σ):
        key1, key2 = jax.random.split(key, 2)

        n_chains = σ.shape[0]
        hilb = sampler.hilbert

        indxs = jax.random.randint(key1, shape=(n_chains,), minval=1, maxval=hilb.size)
        σp, _ = flip_state(hilb, key2, σ, indxs)
        σp = σp.at[:, 0].set(1)

        return σp, None


@struct.dataclass
class Exchange(MetropolisRule):

    neighbors: jax.typing.ArrayLike
    nn_max: int
    nn_dim: jax.typing.ArrayLike
    sigma: float = 1.1  # parameter to adjust neighbor selection

    @override
    def random_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
        key: PRNGKeyT,
    ):

        if sampler.hilbert._total_sz is not None:
            samples = sampler.hilbert.random_state(
                key, size=sampler.n_batches, dtype=sampler.dtype
            )
        else:
            x0 = (-1) ** jnp.arange(sampler.hilbert.size)
            key_set = jax.random.split(key, num=sampler.n_batches)
            samples = jax.vmap(lambda k: jax.random.permutation(k, x0))(key_set).astype(
                sampler.dtype
            )

        return samples

    @override
    def transition(self, sampler, machine, parameters, state, key, σ):

        key1, key2, key3 = jax.random.split(key, (3,))

        n_samples = σ.shape[0]
        N = sampler.hilbert.size

        # Select one spin per sample
        s0 = jax.random.randint(key1, shape=(n_samples,), minval=0, maxval=N)

        # Select one neighbor group per sample (NN/NN2,... etc.)
        nn_rand = jnp.abs(
            self.sigma * jax.random.normal(key2, shape=(n_samples,))
        ).astype(jnp.int32)
        nn_rand = jnp.where(nn_rand < self.nn_max, nn_rand, self.nn_max - 1)

        # Select one neighbor within the group per sample
        keys = jax.random.split(key3, n_samples)

        single_rand = lambda k, nn: jax.random.randint(k, (), 0, self.nn_dim[nn])
        nn_at_rand = jax.vmap(single_rand)(keys, nn_rand)

        s1 = self.neighbors[s0, nn_rand, nn_at_rand]

        # Swap s0 and s1 values in each sample
        mask_s0 = jnp.arange(N) == s0[:, None]
        mask_s1 = jnp.arange(N) == s1[:, None]

        vals_0 = jnp.take_along_axis(σ, indices=s0[:, None], axis=-1)
        vals_1 = jnp.take_along_axis(σ, indices=s1[:, None], axis=-1)

        σ_new = jnp.where(mask_s0, vals_1, jnp.where(mask_s1, vals_0, σ)).astype(
            jnp.int8
        )

        return σ_new, None

    def __repr__(self):
        return "Exchange()"


@struct.dataclass
class ExchangeJ1J2(MetropolisRule):

    J1: float
    J2: float
    J1_nn: jax.typing.ArrayLike
    J2_nn: jax.typing.ArrayLike

    @override
    def random_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
        key: PRNGKeyT,
    ):

        x0 = (-1) ** jnp.arange(sampler.hilbert.size)
        key_set = jax.random.split(key, num=sampler.n_batches)
        samples = jnp.array(
            [jax.random.permutation(key, x0) for key in key_set], dtype=sampler.dtype
        )
        return samples

    @override
    def transition(self, sampler, machine, parameters, state, key, σ):

        key1, key2, key3 = jax.random.split(key, (3,))

        n_samples = σ.shape[0]
        N = sampler.hilbert.size

        # Select one spin per sample
        s0 = jax.random.randint(key1, shape=(n_samples,), minval=0, maxval=N)

        # Probabibility ratio and exchange interaction (select NN or NN2)
        p_J1 = self.J1 / (self.J1 + self.J2)
        is_J1 = jax.random.uniform(key2, shape=(n_samples,)) < p_J1

        # Select one direction (select 1 of d directions)
        key3a, key3b = jax.random.split(key3)
        J1_choices = jax.random.choice(
            key3a, jnp.arange(self.J1_nn.shape[-1]), shape=(n_samples,)
        )
        J2_choices = jax.random.choice(
            key3b, jnp.arange(self.J2_nn.shape[-1]), shape=(n_samples,)
        )
        s1_j1 = self.J1_nn[s0, J1_choices] * is_J1
        s1_j2 = self.J2_nn[s0, J2_choices] * (~is_J1)
        s1 = s1_j1 + s1_j2

        # Swap s0 and s1 values in each sample
        mask_s0 = jnp.arange(N) == s0[:, None]
        mask_s1 = jnp.arange(N) == s1[:, None]

        vals_0 = jnp.take_along_axis(σ, indices=s0[:, None], axis=-1)
        vals_1 = jnp.take_along_axis(σ, indices=s1[:, None], axis=-1)

        σ_new = jnp.where(mask_s0, vals_1, jnp.where(mask_s1, vals_0, σ)).astype(
            jnp.int8
        )

        return σ_new, None


class MultipleRules(netket.sampler.rules.MultipleRules):

    initial_state: str | None = None

    def __init__(self, rules, prules, initial_state=None):
        super().__init__(rules, prules)
        self.initial_state = initial_state

    def random_state(
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
    ):

        if self.initial_state == "M0":
            return M0_state(sampler, machine, params, sampler_state, key)
        elif self.initial_state == "Mmax":
            return Mmax_state(sampler, machine, params, sampler_state, key)
        elif self.initial_state == "hilbert":
            return hilbert_state(sampler, machine, params, sampler_state, key)
        else:
            return super().random_state(sampler, machine, params, sampler_state, key)
