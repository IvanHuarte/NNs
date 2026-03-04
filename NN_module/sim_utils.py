import numpy as np
import jax.numpy as jnp

from NN_module.observables import (
    calc_all_observables_vs,
    calc_all_observables_ED,
    modphase,
)


def measureNdump(keeper, time_exe, exact_diag=False, S_operators=False):

    s_factor = 1 if S_operators else 4

    E_ED = keeper.E_ED / s_factor if hasattr(keeper, "E_ED") else None
    x_ED = keeper.x_ED if hasattr(keeper, "x_ED") else None

    vstate = keeper.best_state
    N = vstate.hilbert.size

    best_step = keeper.best_step
    E_best = float(keeper.best_state_energy) / s_factor
    E_best_per_site = E_best / N
    vscore = s_factor * float(keeper.best_state_vscore)

    modphase_results = {}
    if exact_diag:
        error = float(np.abs(E_best - E_ED) / np.abs(E_ED))
        mp_array_ED, stats_ED = modphase(x_ED)
        modphase_results["xED"] = stats_ED
        print(
            f"xED phase: {stats_ED['phase']['mean']} \u00b1 {stats_ED['phase']['std']}  ({stats_ED['type']})"
        )

    else:
        error = None
        mp_array_ED = None

    mp_array_vs, stats_vs = modphase(vstate)
    modphase_results["vstate"] = stats_vs
    print(
        f"vstate phase: {stats_vs['phase']['mean']} \u00b1 {stats_vs['phase']['std']}  ({stats_vs['type']})"
    )

    # Fidelity
    fidelity = None
    fidelity_per_site = None
    if exact_diag:
        try:
            fidelity = float(jnp.abs(jnp.vdot(vstate.to_array(), x_ED.squeeze())))
            fidelity_per_site = float(jnp.exp(jnp.log(fidelity) / N))
            print(f"Fidelity: {fidelity:.3e}")
        except (MemoryError, RuntimeError, ValueError):
            print(f"Failed fidelity calculation due to memory allocation error")

    # Renyi entropy, magnetization and its fluctuation
    S_renyi, m, ms, m2, ms2 = calc_all_observables_vs(vstate)
    m /= s_factor
    ms /= s_factor
    m2 /= s_factor**2
    ms2 /= s_factor**2

    print(f"\nRenyi entropy: {S_renyi}")
    print(f"< m >: {m}   < m2 >: {m2}")
    print(f"< ms >: {ms}  < ms2 >: {ms2}")

    if exact_diag and vstate.hilbert._total_sz is None:
        m_ED, ms_ED, m2_ED, ms2_ED = calc_all_observables_ED(x_ED.squeeze())
        print(f"ED < m >: {m_ED}   < m2 >: {m2_ED}")
        print(f"ED < ms >: {ms_ED}  < ms2 >: {ms2_ED}\n")

    else:
        m_ED, ms_ED, m2_ED, ms2_ED = None, None, None, None

    results = {
        "best_step": best_step,
        "E_best": E_best,
        "E_best_per_site": E_best_per_site,
        "E_ED": E_ED,
        "error": error,
        "vscore": vscore,
        "time_exe": time_exe,
        "modphase": modphase_results,
        "fidelity": fidelity,
        "fidelity_per_site": fidelity_per_site,
        "S_renyi": S_renyi,
        "m": m,
        "ms": ms,
        "m2": m2,
        "ms2": ms2,
        "m_ED": m_ED,
        "ms_ED": ms_ED,
        "m2_ED": m2_ED,
        "ms2_ED": ms2_ED,
    }

    return results, mp_array_vs, mp_array_ED
