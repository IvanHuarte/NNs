import numpy as np
import jax.numpy as jnp

from NN_module.observables import (
    calc_all_observables_vs,
    calc_all_observables_ED,
    modphase,
)


def measureNdump(keeper, time_exe, exact_diag=False):

    E_ED = keeper.E_ED
    x_ED = keeper.x_ED

    vstate = keeper.best_state
    best_step = keeper.best_step
    E_best = float(keeper.best_state_energy)
    vscore = float(keeper.best_state_vscore)

    modphase_results = {}
    if exact_diag:
        error = float(np.abs(E_best - E_ED) / np.abs(E_ED))
        mp_array_ED, stats_ED = modphase(x_ED)
        modphase_results["xED"] = stats_ED
        print(
            f"xED phase: {stats_ED['phase']['mean']} \u00b1 {stats_ED['phase']['std']}  ({stats_ED['type']})"
        )

    else:
        E_ED = None
        x_ED = None
        error = None

    mp_array_vs, stats_vs = modphase(vstate)
    modphase_results["vstate"] = stats_vs
    print(
        f"vstate phase: {stats_vs['phase']['mean']} \u00b1 {stats_vs['phase']['std']}  ({stats_vs['type']})"
    )

    # Fidelity
    fidelity = None
    try:
        fidelity = float(jnp.abs(jnp.vdot(vstate.to_array(), x_ED.squeeze())))
        print(f"Fidelity: {fidelity:.3e}")
    except (MemoryError, RuntimeError, ValueError):
        print(f"Failed fidelity calculation due to memory allocation error")

    # Renyi entropy, magnetization and its fluctuation
    S_renyi, m, ms, m2, ms2 = calc_all_observables_vs(vstate)

    print(f"\nRenyi entropy: {S_renyi}")
    print(f"< m >: {m}   < m2 >: {m2}")
    print(f"< ms >: {ms}  < ms2 >: {ms2}")

    if exact_diag:
        m_ED, ms_ED, m2_ED, ms2_ED = calc_all_observables_ED(x_ED.squeeze())
        print(f"ED < m >: {m_ED}   < m2 >: {m2_ED}")
        print(f"ED < ms >: {ms_ED}  < ms2 >: {ms2_ED}\n")

    else:
        m_ED, ms_ED, m2_ED, ms2_ED = None

    results = {
        "best_step": best_step,
        "E_best": E_best,
        "E_ED": E_ED,
        "error": error,
        "vscore": vscore,
        "time_exe": time_exe,
        "modphase": modphase_results,
        "fidelity": fidelity,
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
