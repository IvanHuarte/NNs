"""
Metrics of the SanityMonitor. Each function takes (driver, log_data, ctx)
and returns a flat dict of plain numbers (plus, for the per-module metrics,
a list of module rows). They work for any architecture: per-module numbers
use the positional codes of Hydra (see Hydra.callback.introspection).

Cost per call (N_s = samples per step, P = parameters):

    energy         free     (driver statistics of this step)
    sampling       ~free    (+ one pass over the sample array)
    optimization   ~free    (SR direction already computed by the driver)
    wavefunction   1 forward pass on the samples
    gradients      Jacobian on `n_jacobian` samples + NTK spectrum
                   (O(n_jacobian · P) memory, the expensive one)
    local_energy   local energies of all samples (one extra step's worth)
    magnetization  ~free
"""

from __future__ import annotations

import math

import jax
import numpy as np

from ..introspection import (
    aggregate,
    centred_log_derivatives,
    current_diag_shift,
    leaf_table,
    ntk_spectrum,
    subsample,
)


def _f(x):
    """Python float of a (possibly complex / jax) scalar."""
    try:
        return float(np.real(np.asarray(x)))
    except Exception:  # noqa: BLE001
        return math.nan


def energy_metrics(driver, log_data, ctx):
    st = log_data[driver._loss_name]
    E = np.asarray(st.mean)
    var = _f(st.variance)
    err = _f(st.error_of_mean)
    e = _f(E)
    out = {
        "E": e,
        "E_err": err,
        "variance": var,
        "vscore": ctx["N"] * var / e**2 if e else math.nan,
        "imag_sigmas": abs(float(np.imag(E))) / err if err > 0 else 0.0,
    }
    prev = ctx["history"][-1]["energy"] if ctx["history"] and "energy" in ctx["history"][-1] else None
    if prev is not None:
        dE = e - prev["E"]
        out["dE"] = dE
        out["dE_sigmas"] = dE / math.hypot(err, prev["E_err"]) if err > 0 else 0.0
        out["vscore_ratio"] = out["vscore"] / prev["vscore"] if prev["vscore"] > 0 else math.nan
    if ctx.get("E_ref") is not None:
        out["error_ref"] = abs(e - ctx["E_ref"]) / abs(ctx["E_ref"])
    return out


def sampling_metrics(driver, log_data, ctx):
    st = log_data[driver._loss_name]
    vstate = driver.state
    samples = np.asarray(vstate.samples).reshape(-1, vstate.hilbert.size)
    ns = samples.shape[0]
    tau = _f(getattr(st, "tau_corr", math.nan))
    out = {
        "n_samples": ns,
        "chain_length": int(getattr(vstate, "chain_length", 0) or 0),
        "R_hat": _f(getattr(st, "R_hat", math.nan)),
        "tau_corr": tau,
        "ess_eff": 1.0 / (1.0 + 2.0 * tau) if np.isfinite(tau) else math.nan,
        "unique_frac": len(np.unique(samples, axis=0)) / ns,
        # repeated samples are expected when the sector is small
        "n_states": int(vstate.hilbert.n_states) if vstate.hilbert.is_indexable else None,
    }
    acc = getattr(vstate.sampler_state, "acceptance", None)
    out["acceptance"] = _f(acc) if acc is not None else math.nan
    return out


def magnetization_metrics(driver, log_data, ctx):
    vstate = driver.state
    N = vstate.hilbert.size
    m = np.asarray(vstate.samples).reshape(-1, N).sum(axis=1) / 2
    vals, counts = np.unique(m, return_counts=True)
    return {
        "M_mean": float(m.mean()),
        "M_std": float(m.std()),
        "sectors": int(len(vals)),
        "histogram": {float(v): int(c) for v, c in zip(vals, counts)},
        "constrained": getattr(vstate.hilbert, "_total_sz", None) is not None,
    }


def wavefunction_metrics(driver, log_data, ctx, q_max=4):
    vstate = driver.state
    s = np.asarray(vstate.samples).reshape(-1, vstate.hilbert.size)
    logpsi = np.asarray(vstate.log_value(s))
    re = np.real(logpsi)
    ph = np.imag(logpsi)
    out = {
        "finite": bool(np.all(np.isfinite(logpsi))),
        # spread of log|ψ|² over the (|ψ|²-distributed) samples: a few
        # configurations dominating the state show up as a huge range
        "log_amp_range": float(2 * (re.max() - re.min())),
        "log_amp_std": float(2 * re.std()),
    }
    for q in range(1, q_max + 1):
        out[f"phase_q{q}"] = float(np.abs(np.mean(np.exp(1j * q * ph))))
    return out


def local_energy_metrics(driver, log_data, ctx):
    vstate = driver.state
    le = vstate.local_estimators(driver._ham)
    eloc = np.asarray(getattr(le, "data", le)).ravel()  # LocalEstimators (netket 3.22)
    re = np.real(eloc)
    mu, sd = re.mean(), re.std()
    z = np.abs(re - mu) / sd if sd > 0 else np.zeros_like(re)
    return {
        "finite": bool(np.all(np.isfinite(eloc))),
        "max_z": float(z.max()),
        "frac_5sigma": float(np.mean(z > 5)),
        "kurtosis": float(np.mean(z**4)) if sd > 0 else math.nan,
    }


def optimization_metrics(driver, log_data, ctx):
    params = jax.device_get(driver.state.parameters)
    dp = jax.device_get(getattr(driver, "_dp", None))
    rows = leaf_table(params, dp, prev=ctx.get("prev_params"))
    modules, total = aggregate(rows, ctx["depth"], ctx["codes"])
    out = {
        "theta": total["theta"],
        "dp": total.get("dp", math.nan),
        "rel": total.get("rel", math.nan),
        "rel_step": total.get("rel_step", math.nan),
        "eta": total.get("eta", math.nan),
        "finite": all(m["finite"] for m in modules),
        "diag_shift": current_diag_shift(driver),
        "modules": modules,
    }
    info = getattr(driver, "info", None)
    if isinstance(info, dict):
        for k, v in info.items():
            try:
                out[f"info_{k}"] = _f(v)
            except Exception:  # noqa: BLE001
                pass
    return out


def gradient_metrics(driver, log_data, ctx):
    vstate = driver.state
    s = subsample(vstate.samples, ctx["n_jacobian"], seed=ctx["step"])
    O = centred_log_derivatives(vstate, s, chunk_size=ctx.get("chunk_size"))
    params = jax.device_get(vstate.parameters)
    rows = leaf_table(params, O=O)
    modules, total = aggregate(rows, ctx["depth"], ctx["codes"])
    eig = ntk_spectrum(O)
    ds = current_diag_shift(driver)
    eig = np.clip(eig, 0, None)
    out = {
        "n_jacobian": int(s.shape[0]),
        "o_re": total["o_re"],
        "o_im": total["o_im"],
        "dead": total["dead"],
        "ntk_max": float(eig[0]),
        "ntk_eff_rank": int(np.sum(eig > (ds or 0.0))) if ds is not None else math.nan,
        "ntk_rank_frac": float(np.mean(eig > (ds or 0.0))) if ds is not None else math.nan,
        "ntk_cond": float(eig[0] / ds) if ds else math.nan,
        "diag_shift": ds,
        "modules": modules,
    }
    return out


METRICS = {
    "energy": energy_metrics,
    "sampling": sampling_metrics,
    "optimization": optimization_metrics,
    "wavefunction": wavefunction_metrics,
    "gradients": gradient_metrics,
    "local_energy": local_energy_metrics,
    "magnetization": magnetization_metrics,
}

# Names of the previous version, still accepted in config0
ALIASES = {"phase": "wavefunction", "sample_histogram": "magnetization"}


def calc_metrics(driver, log_data, ctx, selection):
    out = {}
    for name in selection:
        out[name] = METRICS[name](driver, log_data, ctx)
    return out
