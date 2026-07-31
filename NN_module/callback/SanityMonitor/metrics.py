import os
import warnings

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from NN_module.utils import print_tree

os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "1"


#### HELPING FUNCTIONS USED IN METRIC CALCULATIONS ####


def tree_norm(tree, axis=None):
    """Norma L2 de PyTree"""

    def leaf_norm(leaf):
        return jnp.sqrt(jnp.sum(jnp.square(jnp.abs(leaf)), axis=axis or None))

    return jax.tree_util.tree_map(leaf_norm, tree)


def tree_global_norm(tree):
    """Total L2 norm of a PyTree"""
    squares_sum = jax.tree_util.tree_map(
        lambda p: jnp.sum(jnp.square(jnp.abs(p))), tree
    )
    return jnp.sqrt(sum(jax.tree_util.tree_leaves(squares_sum)))


def extract_real_imag(tree):
    """Extrae Re/Im de PyTree → DOS PyTrees separados"""

    def get_real(leaf):
        return leaf[:, 0]  # Solo Re

    def get_imag(leaf):
        return leaf[:, 1]  # Solo Im

    O_re = jax.tree_util.tree_map(get_real, tree)
    O_im = jax.tree_util.tree_map(get_imag, tree)
    return O_re, O_im


def percentage_above_threshold(norms_tree, threshold):
    """Calcula el porcentaje de gradientes por muestra que superan un umbral"""
    pct_tree = jax.tree_util.tree_map(lambda x: jnp.mean(x > threshold), norms_tree)
    total_pct = jax.tree_util.tree_reduce(lambda x, y: x + y, pct_tree) * 100
    n_leaves = jax.tree_util.tree_structure(norms_tree).num_leaves
    return total_pct / n_leaves


def percentage_below_threshold(norms_tree, threshold):
    """Calcula el porcentaje de gradientes por muestra que están por debajo de un umbral"""
    pct_tree = jax.tree_util.tree_map(lambda x: jnp.mean(x < threshold), norms_tree)
    total_pct = jax.tree_util.tree_reduce(lambda x, y: x + y, pct_tree) * 100
    n_leaves = jax.tree_util.tree_structure(norms_tree).num_leaves
    return total_pct / n_leaves


### METRICS CALCULATION FUNCTIONS ###


def gradient_metrics(vstate):
    """Compute relevant metrics of the gradients.

    Args:
        params: The parameters of the model.
        apply_fun: The apply function of the model.
        samples: A batch of samples.

    Returns:
        A dictionary containing the mean and standard deviation of the
        gradients' absolute values.
    """
    params = vstate.parameters
    apply_fun = vstate._apply_fun
    samples = vstate.samples
    print("\nSANITY MONITOR\n")

    print_tree(params)

    grad_metrics = {}

    samples = samples.reshape((-1, samples.shape[-1]))
    O = nk.jax.jacobian(apply_fun, params, samples, mode="complex")

    ### 1.- Mean Norm per sample ⟨||∇logψ(s)||⟩ (axis=1 para (Nsamples,...))
    grad_norms_per_sample = tree_norm(O, axis=1)
    mean_norm_sample = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, grad_norms_per_sample)
    ) / len(samples)
    total_norm_std = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(jnp.std, grad_norms_per_sample)
    )
    grad_metrics["norm_per_sample"] = {
        "mean": mean_norm_sample,
        "std": total_norm_std,
    }

    ### 2.- Global Norm ||∇logψ|| normalized
    global_norm = tree_global_norm(O) / jnp.sqrt(len(samples))
    grad_metrics["global_norm"] = global_norm

    ### 3.- Real/Imaginary components stats
    # In global norm
    O_re, O_im = extract_real_imag(O)
    O_real_norm = tree_global_norm(O_re)
    O_imag_norm = tree_global_norm(O_im)

    # In Modulus and Phase nets
    O_modulus = O["ModulusNet"]
    O_mod_re, O_mod_im = extract_real_imag(O_modulus)
    norms_mod_re = tree_norm(O_mod_re, axis=0)  # Normas por muestra
    norms_mod_im = tree_norm(O_mod_im, axis=0)
    O_mod_re_total = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_mod_re)
    )
    O_mod_im_total = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_mod_im)
    )

    try:
        O_phase = O["PhaseNet"]
        O_ph_re, O_ph_im = extract_real_imag(O_phase)
        norms_ph_re = tree_norm(O_ph_re, axis=0)
        norms_ph_im = tree_norm(O_ph_im, axis=0)
        O_ph_re_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_ph_re)
        )
        O_ph_im_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_ph_im)
        )
    except KeyError:
        O_ph_re_total, O_ph_im_total = jnp.nan, jnp.nan
        warnings.warn("PhaseNet not present in architecture", UserWarning)

    except (ValueError, TypeError, MemoryError, RuntimeError) as e:
        raise e

    grad_metrics["ReIm_norms"] = {
        "global": {
            "Re": O_real_norm,
            "Im": O_imag_norm,
        },
        "ModulusNet": {
            "Re": O_mod_re_total,
            "Im": O_mod_im_total,
        },
        "PhaseNet": {
            "Re": O_ph_re_total,
            "Im": O_ph_im_total,
        },
    }

    ### 4.- Percentage of leaves in different norm intervals
    threshold_points = jnp.logspace(-4, 3, num=8)
    intermediate = list(  # 1e-4 < x < 1e3
        map(
            lambda threshold: percentage_above_threshold(
                grad_norms_per_sample, threshold
            ),
            threshold_points,
        )
    )
    lower = [100 - intermediate[0]]  # < 1e-4
    upper = [intermediate[-1]]  # > 1e3

    percentaje_per_norm = (
        lower
        + [intermediate[i] - intermediate[i + 1] for i in range(len(intermediate) - 1)]
        + upper
    )

    grad_metrics["percentage_norm_intervals"] = {
        "log_intervals": threshold_points,
        "percentages": percentaje_per_norm,
    }

    return grad_metrics


def sampling_autocorr_metrics(hamiltonian, vstate):
    """Compute autocorrelation time and the effective number of samples in the driver.

    Args:
        driver: The VMC driver.

    Returns:
        A dictionary containing the autocorrelation time statistics.
         - acceptance: The acceptance rate of the sampler.
         - tau_corr: The estimated autocorrelation time. If
                     NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION = "1",
                     this is computed exactly by using FFT methods. Otherwise,
                     the sum of correlation lags between samples is truncated.
         - ESS: The effective sample size (ESS) calculated as N / (1 + 2 * tau_corr),
                where N is the total number of samples.
         - efficiency: The sampling efficiency defined as ESS / N.
    """

    samples = vstate.samples

    # Local energy (energy of each sample)
    Eloc = vstate.local_estimators(hamiltonian)
    Eloc_flat = Eloc.flatten()
    n_samples = Eloc_flat.shape[0]

    stats = nk.stats.statistics(Eloc_flat)

    if hasattr(stats, "tau_corr_max"):
        tau_corr = (
            stats.tau_corr_max if not np.isnan(stats.tau_corr_max) else stats.tau_corr
        )
    else:
        tau_corr = stats.tau_corr

    ESS = n_samples / (1 + 2 * tau_corr)
    efficiency = ESS / n_samples

    acceptance = vstate.sampler_state.acceptance

    acorr_metrics = {
        "acceptance": acceptance,
        "tau_corr": tau_corr,
        "ESS": ESS,
        "efficiency": efficiency,
        "n_samples": n_samples,
    }
    return acorr_metrics


def sample_magnetization_histogram(vstate):
    """Compute the histogram of magnetization values from the samples.

    Args:
        vstate: The variational state containing the samples.
    Returns:
        A dictionary containing the magnetization values and their corresponding counts.
    """

    N = vstate.hilbert.size
    ms = vstate.samples.reshape(-1, N).sum(axis=1) // 2

    counts, m = np.histogram(
        ms,
        bins=np.arange(-N // 2 - 0.5, N // 2 + 1.5, 1),
    )
    magnetization = (m[:-1] + 0.5).astype(int)[::-1]
    counts = counts[::-1]

    histogram = {"magnetization": magnetization, "counts": counts}
    return histogram


def phase_metrics(vstate, q_max=4):
    """Compute relevant metrics of the phase of the wavefunction.

    Args:
        params: The parameters of the model.
        apply_fun: The apply function of the model.
        samples: A batch of samples.

    Returns:
        A dictionary containing the mean and standard deviation of the
        phase values.
    """
    apply_fun = vstate._apply_fun
    samples = vstate.samples

    phase_metrics = {}

    samples = samples.reshape((-1, samples.shape[-1]))

    log_psi = apply_fun(vstate.variables, samples)
    phases = np.imag(log_psi)

    exps = jnp.arange(1, q_max + 1)[:, None]
    phases = jnp.tile(phases, (q_max, 1))

    q_phases = jnp.exp(1j * exps * phases)

    mean_qphase = jnp.abs(jnp.mean(q_phases, axis=-1))  # |⟨e^{qiϕ}⟩|
    std_qphase = jnp.std(q_phases, axis=-1)

    q_dict = {}

    for i in range(1, q_max + 1):
        q_dict[f"q = {i}"] = {"mean": mean_qphase[i - 1], "std": std_qphase[i - 1]}

    phase_metrics = {**q_dict}

    return phase_metrics


### MAIN METRICS CALCULATION FUNCTION ###

metrics_dict = {
    "gradients": gradient_metrics,
    "sampling": sampling_autocorr_metrics,
    "sample_histogram": sample_magnetization_histogram,
    "phase": phase_metrics,
}


def calc_metrics(hamiltonian, vstate, setup_dict):
    """Calculate all the metrics subscripted in setup_dict
    for monitoring the training.
    Args:
        params: The parameters of the model.
        apply_fun: The apply function of the model.
        samples: The current batch of samples.
        setup_dict: A dictionary containing the metrics to calculate
                    in the form "metric_name": True/False.
    Returns:
        A dictionary containing all computed metrics.
    """
    metrics = {}

    for metric_name, to_compute in setup_dict.items():

        if metric_name not in metrics_dict:
            raise NotImplementedError(
                f"Unknown metric '{metric_name}'. Must be one of {list(metrics_dict.keys())}."
            )
        if to_compute:
            metric_func = metrics_dict[metric_name]

            if metric_name == "sampling":
                metrics[metric_name] = metric_func(hamiltonian, vstate)
            else:
                metrics[metric_name] = metric_func(vstate)

    return metrics
