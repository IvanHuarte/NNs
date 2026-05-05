from NN_module import sampler
import flax.linen as nn
import jax
import jax.numpy as jnp
from netket.utils.types import PyTree, PRNGKeyT


def hilbert_state(
    sampler: "sampler.MetropolisSampler",  # noqa: F821
    machine: nn.Module,
    params: PyTree,
    sampler_state: "sampler.SamplerState",  # noqa: F821
    key: PRNGKeyT,
):
    """
    Initial state function that returns a batch of random states
    chosen randomly from the Hilbert space. This leads in a gaussian
    probability distribution over the magnetization sectors,
    with its maximum at the zero magnetization sector.
    """

    return sampler.hilbert.random_state(
        key, size=sampler.n_batches, dtype=sampler.dtype
    )


def M0_state(
    sampler: "sampler.MetropolisSampler",  # noqa: F821
    machine: nn.Module,
    params: PyTree,
    sampler_state: "sampler.SamplerState",  # noqa: F821
    key: PRNGKeyT,
):
    """
    Initial state function that returns the state of the Hilbert space.
    Random set from M = 0 magnetization sector.
    """

    x0 = (-1) ** jnp.arange(sampler.hilbert.size)
    key_set = jax.random.split(key, num=sampler.n_batches)
    samples = jax.vmap(lambda k: jax.random.permutation(k, x0))(key_set).astype(
        sampler.dtype
    )
    return samples


def Mmax_state(
    sampler: "sampler.MetropolisSampler",  # noqa: F821
    machine: nn.Module,
    params: PyTree,
    sampler_state: "sampler.SamplerState",  # noqa: F821
    key: PRNGKeyT,
):
    """
    Initial state function that returns the state of the Hilbert space.
    Random set from M = N magnetization sector.
    """
    key_set = jax.random.split(key, num=sampler.n_batches)

    x0 = jax.vmap(
        lambda key: jax.random.choice(
            key, a=jnp.array([1, -1], dtype=jnp.int8), shape=(1,)
        )
    )(key_set)
    samples = jnp.repeat(x0, repeats=sampler.hilbert.size, axis=1)

    return samples


def hilbert_even_M_state(
    sampler: "sampler.MetropolisSampler",  # noqa: F821
    machine: nn.Module,
    params: PyTree,
    sampler_state: "sampler.SamplerState",  # noqa: F821
    key: PRNGKeyT,
):
    """
    Initial state function that returns a batch of random states
    chosen randomly from the Hilbert space. This leads in a gaussian
    probability distribution over the magnetization sectors,
    with its maximum at the zero magnetization sector.
    """

    x0 = sampler.hilbert.random_state(
        key, size=sampler.n_batches, dtype=sampler.dtype, even_M=True
    )
    return x0
