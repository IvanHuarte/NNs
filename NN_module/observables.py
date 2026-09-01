import itertools
import sys

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from netket.operator.spin import sigmaz

# Vstate calculations


def renyi_vs(vstate):
    """
    Renyi's entropy
    """
    N = vstate.hilbert.size
    renyi = nkx.observable.Renyi2EntanglementEntropy(
        vstate.hilbert, np.arange(0, N / 2 + 1, dtype=int)
    )
    return vstate.expect(renyi).mean


def M_vs(vstate):
    """
    First magnetization momentum (Ferro)
    """
    N = vstate.hilbert.size
    magn = sum([sigmaz(vstate.hilbert, i) / N for i in range(N)])
    return vstate.expect(magn).mean.real


def M2_vs(vstate):
    """
    Second magnetization momentum (Ferro)
    """
    N = vstate.hilbert.size
    magn2 = sum([sigmaz(vstate.hilbert, i) / N for i in range(N)])
    return vstate.expect(magn2 @ magn2).mean.real


def Ms_vs(vstate):
    """
    First staggered magnetization momentum (AntiFerro)
    """
    N = vstate.hilbert.size
    magn_s = sum([(-1) ** i * sigmaz(vstate.hilbert, i) / N for i in range(N)])
    return vstate.expect(magn_s).mean.real


def Ms2_vs(vstate):
    """
    Second staggered magnetization momentum (AntiFerro)
    """
    N = vstate.hilbert.size
    magn_s2 = sum([(-1) ** i * sigmaz(vstate.hilbert, i) / N for i in range(N)])
    return vstate.expect(magn_s2 @ magn_s2).mean.real


def calc_all_observables_vs(vstate):
    if vstate.hilbert._total_sz is None:
        S_renyi = float(renyi_vs(vstate))
    else:
        S_renyi = None
    m = M_vs(vstate)
    ms = Ms_vs(vstate)
    m2 = M2_vs(vstate)
    ms2 = Ms2_vs(vstate)

    return S_renyi, float(m), float(ms), float(m2), float(ms2)


# Exact Diagonalization calculations


def compute_spin_matrices(N):
    configs = np.arange(2**N)[:, None]
    bits = (configs >> np.arange(N)[::-1]) & 1
    Sz = 0.5 - bits
    return Sz


def M_ED(state):
    """<M> magnetización media"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state) ** 2
    M_exp = np.sum(probs[:, None] * sigma_z, axis=0).mean()
    return float(M_exp)


def M2_ED(state):
    """<M^2> segundo momento de la magnetización uniforme"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state) ** 2

    # correladores S_i S_j
    corr = (probs[:, None, None] * (sigma_z[:, :, None] * sigma_z[:, None, :])).sum(
        axis=0
    )
    M2_exp = corr.sum() / (N**2)
    return float(M2_exp)


def Ms_ED(state):
    """<Ms> magnetización staggered media por sitio"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    staggered = (-1) ** np.arange(N)
    probs = np.abs(state) ** 2
    Ms_exp = np.sum(probs[:, None] * (sigma_z * staggered), axis=0).mean()
    return float(Ms_exp)


def Ms2_ED(state):
    """<Ms^2> segundo momento de la magnetización staggered"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state) ** 2
    staggered = (-1) ** np.arange(N)

    sigma_z_stag = sigma_z * staggered  # shape (2^N, N)
    corr = (
        probs[:, None, None] * (sigma_z_stag[:, :, None] * sigma_z_stag[:, None, :])
    ).sum(axis=0)
    Ms2_exp = corr.sum() / (N**2)
    return float(Ms2_exp)


def calc_all_observables_ED(state):

    N = int(np.log2(len(state)))
    staggered = (-1) ** np.arange(N)

    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state) ** 2

    m = np.sum(probs[:, None] * sigma_z, axis=0).mean()
    ms = np.sum(probs[:, None] * (sigma_z * staggered), axis=0).mean()

    corr = (probs[:, None, None] * (sigma_z[:, :, None] * sigma_z[:, None, :])).sum(
        axis=0
    )
    m2 = corr.sum() / (N**2)

    sigma_z_stag = sigma_z * staggered
    corr = (
        probs[:, None, None] * (sigma_z_stag[:, :, None] * sigma_z_stag[:, None, :])
    ).sum(axis=0)
    ms2 = corr.sum() / (N**2)

    return float(m), float(ms), float(m2), float(ms2)


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


def full_basis_state(x, hi_sub):
    """
    Recives a state `x` in a certain magnetization basis set by `hi_sub`, and
    returns the computational (full) basis state order
    """
    hi_full = nk.hilbert.Spin(s=hi_sub._s, N=hi_sub.size, total_sz=None)
    states_sub = hi_sub.all_states()
    idx_map = hi_full.states_to_numbers(states_sub).reshape(x.shape)
    x_full = jnp.zeros((hi_full.n_states), dtype=x.dtype)
    x_full = x_full.at[idx_map].set(x)

    return x_full


def full2red_basis_idx(hi_sub, indices_full):
    """
    Recives indices in the full basis and returns the corresponding indices
    in the reduced basis set by `hi_sub`
    """
    hi_full = nk.hilbert.Spin(s=hi_sub._s, N=hi_sub.size, total_sz=None)
    states_sub = hi_sub.all_states()
    idx_map = hi_full.states_to_numbers(states_sub)

    inv_map = jnp.full((hi_full.n_states,), -1, dtype=jnp.int32)
    inv_map = inv_map.at[idx_map].set(jnp.arange(hi_sub.n_states, dtype=jnp.int32))

    indices_sub = inv_map[indices_full]

    return indices_sub


def modphase_extended(mod, phase, no_null_mod=True, sigmas=1):
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

    # Histograma solo con las fases correspondientes a módulos no nulos
    if no_null_mod:
        no_null_mod_mask = mod > 1e-12
        phase_hist = phase[no_null_mod_mask]
    else:
        phase_hist = phase

    counts, values = jnp.histogram(phase_hist, bins=1000000)
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
            # if xvs.hilbert._total_sz is not None:
            #     x = full_basis_state(x, xvs.hilbert)
            mod = jnp.abs(x)
            phase = jnp.angle(x)
            return modphase_extended(mod, phase)

        except (MemoryError, RuntimeError, ValueError):
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


def print_max_contributors(x, size, N_max=10, return_states=False):
    (mod, ph), _ = modphase(x)
    configs = all_spin_configurations(size[0] * size[1])

    idx = jnp.argsort(mod)[::-1][:N_max]
    max_configs = configs[idx, :]
    max_mods = mod[idx]
    max_phs = ph[idx]

    for i, config, mod, phs in zip(idx, max_configs, max_mods, max_phs):
        tmagn = jnp.sum(config.flatten())
        print(f"Config ({i}): \n{config.reshape(size)}")
        print(f"M = {tmagn}")
        print(f"\nModulus: {mod}")
        print(f"Phase: {phs}\n")

    if return_states:
        return max_configs


def distance_with_neel(size):
    N = jnp.prod(jnp.array(size))
    i, j = jnp.unravel_index(jnp.arange(N), size)
    neel_A = (-1) ** (i + j).reshape(size)
    neel_B = -1.0 * neel_A

    def _compare(x):
        distance_A = jnp.sum(jnp.abs(x - neel_A))
        distance_B = jnp.sum(jnp.abs(x - neel_B))

        if distance_A < distance_B:
            return distance_A, "Neel-A"
        elif distance_B < distance_A:
            return distance_B, "Neel-B"
        else:
            return distance_A, "Same in Neel-A and Neel-B"

    return _compare


def distance_with_stripped(size):
    N = jnp.prod(jnp.array(size))
    stripped_VA = (-1) ** jnp.arange(N).reshape(size)
    stripped_VB = -1.0 * stripped_VA
    stripped_HA = stripped_VA.T
    stripped_HB = -1.0 * stripped_HA
    patterns = [stripped_VA, stripped_VB, stripped_HA, stripped_HB]
    labels = ["VA", "VB", "HA", "HB"]

    def _compare(x):
        distances = jnp.array([jnp.sum(jnp.abs(x - p)) for p in patterns])

        dmin = jnp.min(distances)
        mask = distances == dmin

        equal_labels = [label for label, m in zip(labels, mask) if bool(m)]
        label_out = "Stripped-"

        for s in equal_labels:
            label_out += f"{s}-"
        label_out = label_out[:-1]

        return dmin, label_out

    return _compare


def measureNdump(keeper, time_exe, S_operators=False):

    s_factor = 4 if S_operators else 1

    E_gr_global = (
        keeper.E_gr_global / s_factor if hasattr(keeper, "E_gr_global") else None
    )
    x_ED_global = keeper.x_ED_global if hasattr(keeper, "x_ED_global") else None

    irrep_info = (
        {"irrep": keeper.irrep, "symmetries": keeper.symmetries}
        if hasattr(keeper, "irrep")
        else None
    )

    E_gr_irrep = keeper.E_gr_irrep / s_factor if hasattr(keeper, "E_gr_irrep") else None
    x_ED_irrep = keeper.x_ED_irrep if hasattr(keeper, "x_ED_irrep") else None

    vstate = keeper.best_state
    N = vstate.hilbert.size

    best_step = keeper.best_step
    E_best = float(keeper.best_state_energy) / s_factor
    E_best_per_site = E_best / N
    vscore = float(keeper.best_state_vscore)

    modphase_results = {}
    error_global = None
    mp_array_ED_global = None
    error_irrep = None
    delta_irrep = None
    mp_array_ED_irrep = None

    if E_gr_global is not None:
        error_global = float(np.abs(E_best - E_gr_global) / np.abs(E_gr_global))
        mp_array_ED_global, stats_ED_global = modphase(x_ED_global)
        modphase_results["xED_global"] = stats_ED_global
        print(
            f"xED phase: {stats_ED_global['phase']['mean']} \u00b1 {stats_ED_global['phase']['std']}  ({stats_ED_global['type']})"
        )

        if E_gr_irrep is not None:
            error_irrep = float(np.abs(E_best - E_gr_irrep) / np.abs(E_gr_irrep))
            delta_irrep = float(E_gr_irrep - E_gr_global)
            mp_array_ED_irrep, stats_ED_irrep = modphase(x_ED_irrep)
            modphase_results["xED_irrep"] = stats_ED_irrep
            print(
                f"xED phase: {stats_ED_irrep['phase']['mean']} \u00b1 {stats_ED_irrep['phase']['std']}  ({stats_ED_irrep['type']})"
            )

    mp_array_vs, stats_vs = modphase(vstate)
    modphase_results["vstate"] = stats_vs
    print(
        f"vstate phase: {stats_vs['phase']['mean']} \u00b1 {stats_vs['phase']['std']}  ({stats_vs['type']})"
    )

    # Fidelity
    fidelity_global = None
    fidelity_per_site_global = None
    fidelity_irrep = None
    fidelity_per_site_irrep = None

    if E_gr_global is not None:
        try:
            fidelity_global = float(
                jnp.abs(jnp.vdot(vstate.to_array(), x_ED_global.squeeze()))
            )
            fidelity_per_site_global = float(jnp.exp(jnp.log(fidelity_global) / N))
            print(f"Fidelity of global ground state: {fidelity_global:.3e}")
        except (MemoryError, RuntimeError, ValueError):
            print("Failed global fidelity calculation due to memory allocation error")

        if E_gr_irrep is not None:
            try:
                fidelity_irrep = float(
                    jnp.abs(jnp.vdot(vstate.to_array(), x_ED_irrep.squeeze()))
                )
                fidelity_per_site_irrep = float(jnp.exp(jnp.log(fidelity_irrep) / N))
                print(f"Fidelity of irrep ground state: {fidelity_irrep:.3e}")
            except (MemoryError, RuntimeError, ValueError):
                print(
                    "Failed global fidelity calculation due to memory allocation error"
                )

    # Renyi entropy, magnetization and its fluctuation
    S_renyi, m, ms, m2, ms2 = calc_all_observables_vs(vstate)
    m /= s_factor
    ms /= s_factor
    m2 /= s_factor**2
    ms2 /= s_factor**2

    print(f"\nRenyi entropy: {S_renyi}")
    print(f"< m >: {m}   < m2 >: {m2}")
    print(f"< ms >: {ms}  < ms2 >: {ms2}")

    x_ED = x_ED_global if x_ED_global is not None else None
    x_ED = x_ED_irrep if x_ED_irrep is not None else x_ED

    if x_ED is not None and vstate.hilbert._total_sz is None:
        m_ED, ms_ED, m2_ED, ms2_ED = calc_all_observables_ED(x_ED.squeeze())
        print(f"ED < m >: {m_ED}   < m2 >: {m2_ED}")
        print(f"ED < ms >: {ms_ED}  < ms2 >: {ms2_ED}\n")

    else:
        m_ED, ms_ED, m2_ED, ms2_ED = None, None, None, None

    results = {
        "best_step": best_step,
        "E_best": E_best,
        "E_best_per_site": E_best_per_site,
        "E_gr_global": E_gr_global,
        "E_gr_irrep": E_gr_irrep,
        "error_global": error_global,
        "error_irrep": error_irrep,
        "delta_irrep": delta_irrep,
        "vscore": vscore,
        "time_exe": time_exe,
        "modphase": modphase_results,
        "fidelity_global": fidelity_global,
        "fidelity_per_site_global": fidelity_per_site_global,
        "fidelity_irrep": fidelity_irrep,
        "fidelity_per_site_irrep": fidelity_per_site_irrep,
        "S_renyi": S_renyi,
        "m": m,
        "ms": ms,
        "m2": m2,
        "ms2": ms2,
        "m_ED": m_ED,
        "ms_ED": ms_ED,
        "m2_ED": m2_ED,
        "ms2_ED": ms2_ED,
        "irrep_info": irrep_info,
    }

    return results, mp_array_vs, mp_array_ED_global, mp_array_ED_irrep
