import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import netket as nk
import numpy.typing as npt
from typing import Optional, Tuple
import sys
from pathlib import Path
import itertools

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent
        / "Transformers/transformer_LR_WF_public"
    )
)
from transformer_LR_WF.utils import InvertMagnetization


REAL_DTYPE = jnp.asarray(1.0).dtype

optimizer_name_dict = {
    "Sgd": nk.optimizer.Sgd,
    "adam": nk.optimizer.Adam,
    "AdaGrad": nk.optimizer.AdaGrad,
}

sampler_dict = {
    "MetropolisLocal": nk.sampler.MetropolisLocal,
    "MetropolisExchange": nk.sampler.MetropolisExchange,
    "MetropolisHamiltonian": nk.sampler.MetropolisHamiltonian,
    "MetropolisSampler": nk.sampler.MetropolisSampler,
}

rule_dict = {
    "LocalRule/InvertMagnetization": nk.sampler.rules.MultipleRules(
        [nk.sampler.rules.LocalRule(), InvertMagnetization()], [0.75, 0.25]
    ),
}


def cos_exp_scheduler(epochs, lr0, decay, cycles, n, lr_min):

    exp = lambda step, a: jnp.exp(-a * step / epochs)
    cos = lambda step, cycles: jnp.cos(step * 2 * jnp.pi / epochs * (cycles - 0.5))
    line = lambda step, n: (lr_min - n) / epochs * step + n

    def scheduler_callable(step):
        return (lr0 - n) * exp(step, decay) * (1 + cos(step, cycles)) / 2 + line(
            step, n
        )

    return scheduler_callable


def scheduler_initializer(name, setup):

    epochs = setup["total_epochs"]

    if name == "cos_exp_scheduler":
        return cos_exp_scheduler(
            epochs=epochs,
            lr0=setup["lr0"],
            decay=setup["decay_exp"],
            cycles=setup["cosine_cycles"],
            n=setup["n"],
            lr_min=setup["lr_min"],
        )
    elif name == "warmup_exponential_decay":
        return optax.warmup_exponential_decay_schedule(
            init_value=setup["lr0"],
            peak_value=setup["peak_value"],
            warmup_steps=setup["warmup_steps"],
            transition_steps=1,
            decay_rate=setup["decay_rate"],
        )


def circulant(row: npt.ArrayLike, times: Optional[int] = None) -> npt.ArrayLike:
    """Build a (full or partial) circulant matrix based on an array.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        If `times` is None, a square matrix with all the offset versions of the
        first argument. Otherwise, `times` rows of a circulant matrix.
    """
    row = jnp.asarray(row)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1)
        return (new_carry, new_carry)

    if times is None:
        nruter = jax.lax.scan(scan_arg, row, row)[1][::-1, :]
    else:
        nruter = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return nruter


def traslations_2D_vmap(
    x: npt.ArrayLike, size: Tuple[int, int] = (1, 1)
) -> npt.ArrayLike:
    """
    Returns a matrix of translations of a flat input vector x
    which represtents a 2D lattice state.
    Enhances time execution.
    """
    x = x.reshape(-1, *size)
    N = size[0] * size[1]
    idx = jnp.unravel_index(jnp.arange(N), size)
    shift_idx = jnp.array(idx).T

    roll = lambda x, shift: jnp.roll(x, shift=shift, axis=(-2, -1)).reshape(-1, N)

    return jax.vmap(roll, in_axes=(None, 0))(x, shift_idx).reshape(-1, N).squeeze()


def traslations_2D_scan(x: npt.ArrayLike, size: Tuple[int, int]) -> npt.ArrayLike:
    """
    Returns a matrix of translations of a flat input vector x
    which represtents a 2D lattice state.
    Enhances memory saving.
    N must be equal to x.shape[-1]
    """
    N = size[0] * size[1]
    x = x.reshape(size)

    def scan_and_roll_x(carry_x, _):

        def scan_and_roll_y(carry_y, _):
            y = jnp.roll(carry_y, shift=1, axis=-1)
            return y, y.reshape(-1, N)

        _, block_y = jax.lax.scan(scan_and_roll_y, carry_x, length=size[1])
        x = jnp.roll(carry_x, shift=1, axis=-2)

        return x, block_y

    return jax.lax.scan(scan_and_roll_x, x, length=size[0])[1].reshape(-1, N).squeeze()


def traslations_2D(
    x: npt.ArrayLike,
    size: Tuple[int, int],
    token_size: Tuple[int, int] = None,
    memory: bool = False,
) -> npt.ArrayLike:
    """
    Expects a x=(B,N) tensor and returns a x=(N_tr, B, N_to), where
    N_tr is the number of traslations in the group and N_to is equal to
    N (token_size=None) or a tuple showing a tokenized lattice
    (N_tokens,token_dim) (token_size!=None)
    """

    x = jnp.atleast_2d(x)
    B = x.shape[0]

    if token_size is None:
        token_size = size

    if x.shape[1] != size[0] * size[1]:
        raise ValueError(
            "`x` dimension must be equal to `prod(size)`, "
            + f"but got {x.shape[0]} and {size[0] * size[1]}."
        )

    if memory:
        x = traslations_2D_scan(x, size)
    else:
        x = traslations_2D_vmap(x, size)

    sub_lat = (size[0] // token_size[0], size[1] // token_size[1])

    # Transforms x into different shapes depending on token_size
    x = (
        x.reshape((-1, sub_lat[1], token_size[1], sub_lat[0], token_size[0]), order="C")
        .transpose((0, 1, 3, 2, 4))
        .reshape(size[0] * size[1], B, -1, token_size[0] * token_size[1])
        .squeeze()
    )

    return x


def phase_stats_vstate(vstate, eps=0.01):

    samples = vstate.samples
    flat_samples = samples.reshape(-1, samples.shape[-1])
    logpsi = vstate.log_value(flat_samples)
    phases = jnp.imag(logpsi)

    mean = float(phases.mean(axis=0))
    std = float(jnp.std(phases))
    if std > eps:
        psi = "complex"
    else:
        psi = "real"

    return mean, std, psi


def phase_stats_ED(x_ED, eps=0.1):

    phases = jnp.angle(x_ED)

    mean = float(phases.mean(axis=0)[0])
    std = float(jnp.std(phases))
    if std > eps:
        psi = "complex"
    else:
        psi = "real"

    return mean, std, psi


def modphase_extended(mod, phase, sigmas=1):
    """
    Calculates modulus and phase for a expanded hilbert vector. Returns also statics for both
    modulus and phase and detects symmetry peaks for phase.

    Input:
        - x (ArrayLike) Wavefunction (C²)
        - sigmas: Number of sigmas to establish the threshold for values considered peaks

    Return:
        - modulus
        - phase
        - stats: Statistics as the mean and standard deviation for modulus and phase. Phase
                 also includes info about phase peaks. peaks:(angle, counts)
    """
    phase = (phase + jnp.pi) % (2 * jnp.pi) - jnp.pi  # Put on interval[-pi,pi)

    mean_mod = mod.mean()
    std_mod = mod.std()
    mean_phase = phase.mean()
    std_phase = phase.std()

    if std_phase > 0.1:
        psi = "complex"
    else:
        psi = "real"

    stats = {
        "type": psi,
        "modulus": {"mean": float(mean_mod), "std": float(std_mod)},
        "phase": {"mean": float(mean_phase), "std": float(std_phase)},
    }

    counts, values = jnp.histogram(phase, bins=1000000)
    nonzero_mask = jnp.where(counts != 0.0)[0]
    nonzero_counts, nonzero_values = counts[nonzero_mask], values[nonzero_mask]
    mean = nonzero_counts.mean()
    std = nonzero_counts.std()

    if std > mean:

        peaks_idx = jnp.where(nonzero_counts > mean + sigmas * std)
        peaks_x = nonzero_values[peaks_idx]
        counts_x = nonzero_counts[peaks_idx]

        idx = jnp.argsort(peaks_x)
        peaks = peaks_x[idx]
        peak_counts = counts_x[idx]

        stats["peaks"] = {
            "values": [float(p) for p in peaks],
            "counts": [float(pc) for pc in peak_counts],
            "count_mean": float(mean),
            "count_std": float(std),
        }
    else:
        stats["peaks"] = None

    mod_phase = np.array([mod, phase]).squeeze()

    return mod_phase, stats


def modphase(xvs):

    if isinstance(xvs, (np.ndarray, jax.Array)):
        # print("ED in modphase")
        mod = jnp.abs(xvs)
        phase = jnp.angle(xvs)
        return modphase_extended(mod, phase)

    elif isinstance(xvs, nk.vqs.VariationalState):
        try:
            # print("vstate in modphase")
            x = xvs.to_array()
            mod = jnp.abs(x)
            phase = jnp.angle(x)
            return modphase_extended(mod, phase)

        except (MemoryError, RuntimeError, ValueError) as error:
            samples = xvs.samples
            flat_samples = samples.reshape(-1, samples.shape[-1])
            logpsi = xvs.log_value(flat_samples)

            mod = jnp.exp(jnp.real(logpsi))
            phase = jnp.imag(logpsi)
            return modphase_extended(mod, phase)

    else:
        print(f"ERROR. Unknown input instance for vstate: {type(xvs)}")
        sys.exit(1)


def all_spin_configurations(N):
    # Genera todas las combinaciones posibles de N spines con valores ±1
    return jnp.array(list(itertools.product([-1, 1], repeat=N)))


def print_max_contributors(x, size, N_max=10):
    (mod, ph), _ = modphase(x)
    configs = all_spin_configurations(size[0] * size[1])

    idx = jnp.argsort(mod)[::-1][:N_max]
    max_configs = configs[idx, :]
    max_mods = mod[idx]
    max_phs = ph[idx]

    for config, mod, phs in zip(max_configs, max_mods, max_phs):
        print(f"Config: \n{config.reshape(size)}")
        print(f"\nModulus: {mod}")
        print(f"Phase: {phs}\n")
