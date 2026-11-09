"""
Generic introspection of a variational state during training, shared by
SanityMonitor and LeavesGradient. Nothing here assumes a particular
architecture: parameters are grouped by the same positional codes used by
the schedule and by Hydra's `load` instructions (Hydra.NN.utils), so a
"module" is any node of the parameter tree at a chosen depth:

    depth 1   first-level submodules (e.g. backbone / head of Sequivariant)
    depth 2   their children (e.g. the StageBlocks of a CvT3)
    ...

Quantities (all per module, computed on host-small arrays):

    |θ|          RMS of the parameters
    |dp|         RMS of the update proposed by the driver (SR direction,
                 driver._dp, before the optimizer applies the learning rate)
    |dp|/|θ|     relative size of the SR direction (before the learning rate)
    share        fraction of the global |dp|² carried by the module
    |Δθ|/|θ|     relative size of the step actually applied (θ_t − θ_{t−1}:
                 learning rate, momentum and schedule masks included)
    |O_re|, |O_im|
                 RMS over samples and parameters of the centred log-derivatives
                 ∂Re logψ/∂θ (modulus) and ∂Im logψ/∂θ (phase), on a subsample
    dead         fraction of parameters whose log-derivative has (numerically)
                 zero variance over the subsample: they cannot be trained by SR
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from Hydra.NN.utils import CODE_SKIP_LABELS, get_code2path_flatten

# Labels without a positional code (wrappers) and the flax collection name
SKIP_LABELS = frozenset([*CODE_SKIP_LABELS, "params"])


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _key_name(k):
    return str(getattr(k, "key", getattr(k, "name", k)))


def leaf_paths(tree):
    """[(path tuple of str, leaf)] in tree order."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(tuple(_key_name(k) for k in path), leaf) for path, leaf in flat]


def strip_wrappers(path):
    """Path without wrapper labels (NN, Z2Wrap, TWrap, params...)."""
    return tuple(p for p in path if p not in SKIP_LABELS)


def path_codes(params):
    """{stripped path: code} for every internal node (as in the schedule)."""
    try:
        c2p = get_code2path_flatten(dict(params))
    except Exception:  # noqa: BLE001 - non-dict trees (plain tests)
        return {}
    return {strip_wrappers(p): c for c, p in c2p.items()}


def module_of(path, depth):
    """Module key of a leaf path at the given depth (leaf name excluded)."""
    p = strip_wrappers(path)
    inner = p[:-1]  # last element is the array name (kernel, bias, ...)
    return inner[:depth] if inner else ("<root>",)


def module_label(key, codes):
    code = codes.get(key)
    name = "/".join(key)
    return f"{code} {name}" if code is not None else name


# ---------------------------------------------------------------------------
# Per-leaf / per-module statistics
# ---------------------------------------------------------------------------


def _sumsq(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x) ** 2)), x.size, bool(np.all(np.isfinite(x)))


def leaf_table(params, dp=None, O=None, prev=None, dead_tol=1e-10):
    """
    One row per parameter leaf:
        path, size, ss_theta, ss_dp, ss_step, ss_ore, ss_oim, n_dead, finite
    (ss_* = sum of squares; O = (O_re, O_im) trees of shape (Ns, *leaf);
    prev = parameters of the previous step, for the applied step).
    """
    rows = []
    p_leaves = leaf_paths(params)
    dp_leaves = [l for _, l in leaf_paths(dp)] if dp is not None else [None] * len(p_leaves)
    pv_leaves = [l for _, l in leaf_paths(prev)] if prev is not None else [None] * len(p_leaves)
    if len(pv_leaves) != len(p_leaves):  # architecture changed
        pv_leaves = [None] * len(p_leaves)
    if O is not None:
        ore = [l for _, l in leaf_paths(O[0])]
        oim = [l for _, l in leaf_paths(O[1])]
    else:
        ore = oim = [None] * len(p_leaves)
    for (path, th), d, a, b, pv in zip(p_leaves, dp_leaves, ore, oim, pv_leaves):
        ss_t, size, fin = _sumsq(th)
        row = {"path": path, "size": size, "ss_theta": ss_t, "finite": fin}
        if pv is not None and np.shape(pv) == np.shape(th):
            row["ss_step"] = _sumsq(np.asarray(th) - np.asarray(pv))[0]
        if d is not None:
            ss_d, _, fin_d = _sumsq(d)
            row["ss_dp"], row["finite"] = ss_d, fin and fin_d
        if a is not None:
            a, b = np.asarray(a), np.asarray(b)
            ns = a.shape[0]
            row["ss_ore"] = float(np.sum(a**2)) / ns
            row["ss_oim"] = float(np.sum(b**2)) / ns
            var = (a**2 + b**2).mean(axis=0).ravel()
            row["n_dead"] = int(np.sum(var < dead_tol))
            # a scalar with zero centred derivative is a global constant of
            # log ψ (normalisation / global phase): expected, not "dead"
            row["gauge"] = size == 1 and row["n_dead"] == 1
        rows.append(row)
    return rows


def aggregate(rows, depth, codes=None):
    """Sum leaf rows into module rows at `depth`; returns (modules, totals)."""
    codes = codes or {}
    mods = {}
    for r in rows:
        key = module_of(r["path"], depth)
        m = mods.setdefault(key, {"key": key, "label": module_label(key, codes),
                                  "size": 0, "n_leaves": 0, "finite": True})
        m["size"] += r["size"]
        m["n_leaves"] += 1
        m["finite"] &= r["finite"]
        for k in ("ss_theta", "ss_dp", "ss_step", "ss_ore", "ss_oim", "n_dead"):
            if k in r:
                m[k] = m.get(k, 0) + (0 if (k == "n_dead" and r.get("gauge")) else r[k])
    total = {"size": sum(m["size"] for m in mods.values())}
    for k in ("ss_theta", "ss_dp", "ss_step", "ss_ore", "ss_oim", "n_dead"):
        if any(k in m for m in mods.values()):
            total[k] = sum(m.get(k, 0) for m in mods.values())
    for m in list(mods.values()) + [total]:
        finish_row(m, total)
    return list(mods.values()), total


def finish_row(m, total):
    """RMS values and ratios of an aggregated row (in place)."""
    n = max(m["size"], 1)
    m["theta"] = math.sqrt(m["ss_theta"] / n) if "ss_theta" in m else math.nan
    tiny = m["theta"] < 1e-12  # e.g. biases initialised at zero: ratio undefined
    if "ss_dp" in m:
        m["dp"] = math.sqrt(m["ss_dp"] / n)
        m["rel"] = math.nan if tiny else m["dp"] / m["theta"]
        tot = total.get("ss_dp", 0.0)
        m["share"] = m["ss_dp"] / tot if tot > 0 else math.nan
    if "ss_step" in m:
        m["step"] = math.sqrt(m["ss_step"] / n)
        m["rel_step"] = math.nan if tiny else m["step"] / m["theta"]
        if "dp" in m and m["dp"] > 0:
            m["eta"] = m["step"] / m["dp"]  # effective learning rate
    if "ss_ore" in m:
        m["o_re"] = math.sqrt(m["ss_ore"] / n)
        m["o_im"] = math.sqrt(m["ss_oim"] / n)
        m["dead"] = m["n_dead"] / n
    return m


# ---------------------------------------------------------------------------
# Log-derivatives on a subsample
# ---------------------------------------------------------------------------


def subsample(samples, n, seed=0):
    """At most n samples (flattened), chosen without replacement."""
    s = np.asarray(samples).reshape(-1, samples.shape[-1])
    if n is None or n >= s.shape[0]:
        return jnp.asarray(s)
    idx = np.random.default_rng(seed).choice(s.shape[0], size=n, replace=False)
    return jnp.asarray(s[idx])


def centred_log_derivatives(vstate, samples, chunk_size=None):
    """
    (O_re, O_im): trees of shape (Ns, *leaf) with the centred derivatives of
    Re logψ and Im logψ w.r.t. the (real) parameters.
    """
    O = nk.jax.jacobian(
        vstate._apply_fun, vstate.parameters, samples, vstate.model_state,
        mode="complex", center=True, chunk_size=chunk_size,
    )
    o_re = jax.tree_util.tree_map(lambda x: np.asarray(x[:, 0]), O)
    o_im = jax.tree_util.tree_map(lambda x: np.asarray(x[:, 1]), O)
    return o_re, o_im


def ntk_spectrum(O):
    """
    Eigenvalues of the real NTK  T = A Aᵀ / Ns with A = [O_re; O_im]
    (2Ns x P): its non-zero spectrum is the one of the S matrix that SR inverts.
    """
    o_re, o_im = O
    leaves_re = [np.asarray(x).reshape(x.shape[0], -1) for x in jax.tree_util.tree_leaves(o_re)]
    leaves_im = [np.asarray(x).reshape(x.shape[0], -1) for x in jax.tree_util.tree_leaves(o_im)]
    a = np.concatenate([np.concatenate(leaves_re, 1), np.concatenate(leaves_im, 1)], 0)
    ns = a.shape[0] // 2
    t = (a @ a.T) / ns
    return np.sort(np.linalg.eigvalsh(t))[::-1]


def current_diag_shift(driver):
    ds = getattr(driver, "diag_shift", None)
    if ds is None:
        return None
    try:
        return float(ds(driver.step_count)) if callable(ds) else float(ds)
    except Exception:  # noqa: BLE001
        return None
