"""
Diagnosis of the SanityMonitor metrics. Every metric becomes a *check*

    {"name", "value" (str), "status" in STATUSES, "message"}

grouped by category. Thresholds are deliberately conservative: "warn" means
"look at it", "error" means the run is very likely broken. Categories with
per-module data also carry a "modules" table.
"""

from __future__ import annotations

import math

STATUSES = ("ok", "info", "warn", "error")
RANK = {s: i for i, s in enumerate(STATUSES)}

# thresholds (one place to tune them)
TH = {
    "imag_sigmas": 5.0,  # |Im E| / error of the mean
    "dE_up_sigmas": 3.0,  # energy rise since the previous check
    "dE_up_sigmas_error": 10.0,
    "vscore_jump": 3.0,  # V-score ratio to the previous check
    "acc_low": 0.15,
    "acc_frozen": 0.05,
    "acc_high": 0.95,
    "min_chain": 32,  # samples per chain needed for R̂ / τ_corr
    "rhat_warn": 1.1,
    "rhat_error": 1.5,
    "tau_warn": 5.0,
    "tau_error": 20.0,
    "unique_warn": 0.2,
    "unique_error": 0.05,
    "rel_step_warn": 0.05,  # |Δθ|/|θ| of the applied step
    "rel_step_error": 0.5,
    "module_rel_outlier": 100.0,  # module |Δθ|/|θ| vs median of modules
    "dead_warn": 0.2,
    "dead_error": 0.5,
    "ntk_rank_frac": 0.05,
    "ntk_cond": 1e10,
    "log_amp_std": 10.0,
    "eloc_outliers": 0.01,
    "eloc_kurtosis": 50.0,
}


def _c(name, value, status="ok", message=""):
    return {"name": name, "value": value, "status": status, "message": message}


def _fmt(x, f="{:.3e}"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f.format(x)


def diag_energy(m):
    ch = [_c("E", f"{m['E']:.6f} ± {m['E_err']:.1e}", "info")]
    st = "error" if not math.isfinite(m["vscore"]) else "info"
    ch.append(_c("V-score", _fmt(m["vscore"]), st,
                 "Energía o varianza no finitas" if st == "error" else "N·Var/E²: calidad intensiva del estado"))
    if m["imag_sigmas"] > TH["imag_sigmas"]:
        ch.append(_c("Im E", f"{m['imag_sigmas']:.1f} σ", "warn",
                     "Parte imaginaria de E significativa: ruido de la fase o muestras insuficientes"))
    else:
        ch.append(_c("Im E", f"{m['imag_sigmas']:.1f} σ", "ok", "Compatible con cero"))
    if "dE" in m:
        s = m["dE_sigmas"]
        st = "error" if s > TH["dE_up_sigmas_error"] else "warn" if s > TH["dE_up_sigmas"] else "ok"
        msg = {"ok": "Evolución normal desde la última comprobación",
               "warn": "La energía ha subido de forma significativa",
               "error": "Subida fuerte de la energía: posible inestabilidad (lr / diag_shift)"}[st]
        ch.append(_c("ΔE", f"{m['dE']:+.4f} ({s:+.1f} σ)", st, msg))
        r = m["vscore_ratio"]
        st = "warn" if math.isfinite(r) and r > TH["vscore_jump"] else "ok"
        ch.append(_c("V-score / anterior", _fmt(r, "{:.2f}"), st,
                     "El V-score se ha multiplicado: el estado empeora" if st == "warn" else ""))
    if "error_ref" in m:
        ch.append(_c("Error vs ED", _fmt(m["error_ref"], "{:.3e}"), "info"))
    return {"checks": ch}


def diag_sampling(m):
    ch = []
    a = m["acceptance"]
    if not math.isfinite(a):
        ch.append(_c("Aceptación", "n/a", "info"))
    elif a < TH["acc_frozen"]:
        ch.append(_c("Aceptación", f"{a:.3f}", "error", "Cadenas congeladas: casi ninguna propuesta se acepta"))
    elif a < TH["acc_low"]:
        ch.append(_c("Aceptación", f"{a:.3f}", "warn", "Aceptación baja: mezcla lenta"))
    elif a > TH["acc_high"]:
        ch.append(_c("Aceptación", f"{a:.3f}", "warn", "Aceptación casi total: |ψ| casi plana o propuestas triviales"))
    else:
        ch.append(_c("Aceptación", f"{a:.3f}", "ok"))
    short = m.get("chain_length", 0) < TH["min_chain"]  # R̂ / τ unreliable on short chains
    r = m["R_hat"]
    st = ("info" if (short or not math.isfinite(r)) else
          "error" if r > TH["rhat_error"] else "warn" if r > TH["rhat_warn"] else "ok")
    ch.append(_c("R̂", _fmt(r, "{:.3f}"), st, {
        "warn": "Cadenas no mezcladas del todo (R̂ > 1.1)",
        "error": "Cadenas en regiones distintas: estimaciones sesgadas",
        "info": f"Poco fiable con cadenas de {m.get('chain_length', 0)} muestras"}.get(st, "")))
    t = m["tau_corr"]
    st = ("info" if (short or not math.isfinite(t)) else
          "error" if t > TH["tau_error"] else "warn" if t > TH["tau_warn"] else "ok")
    ch.append(_c("τ_corr", _fmt(t, "{:.2f}"), st, {
        "warn": "Autocorrelación apreciable", "error": "Autocorrelación alta: muestras poco independientes",
        "info": "Poco fiable con cadenas cortas"}.get(st, "")))
    ch.append(_c("ESS / N", _fmt(m["ess_eff"], "{:.2f}"), "info"))
    u = m["unique_frac"]
    small = m.get("n_states") is not None and m["n_states"] < 20 * m["n_samples"]
    st = "info" if small else "error" if u < TH["unique_error"] else "warn" if u < TH["unique_warn"] else "ok"
    ch.append(_c("Muestras distintas", f"{u:.2f}", st, {
        "warn": "Muchas muestras repetidas: estado concentrado o sampler atascado",
        "error": "Casi todas las muestras iguales: sampler atascado"}.get(st, "")))
    return {"checks": ch}


def diag_magnetization(m):
    msg = "Sector fijo (sz_total)" if m["constrained"] else f"{m['sectors']} sectores visitados"
    hist = ", ".join(f"{int(k)}:{v}" for k, v in sorted(m["histogram"].items()))
    return {"checks": [_c("M", f"{m['M_mean']:+.2f} ± {m['M_std']:.2f}", "info", msg),
                       _c("Histograma M", hist if len(hist) < 120 else hist[:117] + "...", "info")]}


def diag_wavefunction(m):
    ch = []
    if not m["finite"]:
        ch.append(_c("log ψ", "NaN/Inf", "error", "Valores no finitos de log ψ en las muestras"))
    s = m["log_amp_std"]
    st = "warn" if s > TH["log_amp_std"] else "ok"
    ch.append(_c("σ[log|ψ|²]", f"{s:.2f}  (rango {m['log_amp_range']:.1f})", st,
                 "Amplitudes muy dispares entre muestras: pocas configuraciones dominan" if st == "warn" else ""))
    q = [m[f"phase_q{i}"] for i in range(1, 5) if f"phase_q{i}" in m]
    if q:
        kind = ("fase casi constante" if q[0] > 0.9 else
                "signo ±1 (estructura de Marshall)" if len(q) > 1 and q[1] > 0.9 else
                "fase con estructura" if max(q) > 0.3 else "fase dispersa")
        ch.append(_c("|⟨e^{iqφ}⟩| q=1..4", "  ".join(f"{x:.2f}" for x in q), "info", kind))
    return {"checks": ch}


def diag_local_energy(m):
    ch = []
    if not m["finite"]:
        ch.append(_c("E_loc", "NaN/Inf", "error", "Energías locales no finitas"))
    f = m["frac_5sigma"]
    st = "warn" if f > TH["eloc_outliers"] else "ok"
    ch.append(_c("E_loc > 5σ", f"{100 * f:.2f} %  (max {m['max_z']:.1f} σ)", st,
                 "Colas pesadas: estimador ruidoso (nodos de ψ o muestras raras)" if st == "warn" else ""))
    k = m["kurtosis"]
    st = "warn" if math.isfinite(k) and k > TH["eloc_kurtosis"] else "ok"
    ch.append(_c("Curtosis E_loc", _fmt(k, "{:.1f}"), st))
    return {"checks": ch}


def diag_optimization(m):
    ch = []
    if not m["finite"]:
        ch.append(_c("Parámetros / dp", "NaN/Inf", "error", "Parámetros o actualización no finitos"))
    r = m["rel_step"]
    if math.isfinite(r):
        st = "error" if r > TH["rel_step_error"] else "warn" if r > TH["rel_step_warn"] else "ok"
        ch.append(_c("|Δθ| / |θ| (paso aplicado)", _fmt(r, "{:.2e}"), st, {
            "ok": "Tamaño relativo del último paso (lr, momentum y máscaras incluidos)",
            "warn": "Pasos grandes: la red cambia mucho en cada paso (reducir lr)",
            "error": "Paso enorme: los parámetros se reescriben en cada paso"}[st]))
        ch.append(_c("lr efectivo |Δθ|/|dp|", _fmt(m["eta"], "{:.2e}"), "info",
                     "Debe coincidir con el lr del schedule (salvo momentum o máscaras)"))
    ch.append(_c("|dp| / |θ| (dirección SR)", f"{_fmt(m['dp'])} / {_fmt(m['theta'])} = {_fmt(m['rel'], '{:.2e}')}",
                 "info", "Tamaño de la dirección SR antes del lr"))
    if m.get("diag_shift") is not None:
        ch.append(_c("diag_shift", _fmt(m["diag_shift"]), "info"))
    mods = m["modules"]
    key = "rel_step" if any(math.isfinite(x.get("rel_step", math.nan)) for x in mods) else "rel"
    rels = sorted(x[key] for x in mods if math.isfinite(x.get(key, math.nan)) and x[key] > 0)
    if len(rels) > 1:
        med = rels[len(rels) // 2]
        for x in mods:
            v = x.get(key, math.nan)
            if math.isfinite(v) and v > TH["module_rel_outlier"] * med:
                ch.append(_c(f"Módulo {x['label']}", f"{v:.1e}", "warn",
                             f"Se mueve {v / med:.0f}× más que la mediana de los módulos"))
            if x.get("ss_dp", 1.0) == 0.0:
                ch.append(_c(f"Módulo {x['label']}", "dp = 0", "info", "Congelado (modo del schedule)"))
    return {"checks": ch, "modules": mods}


def diag_gradients(m):
    ch = []
    d = m["dead"]
    st = "error" if d > TH["dead_error"] else "warn" if d > TH["dead_warn"] else "ok"
    ch.append(_c("Parámetros muertos", f"{100 * d:.1f} %", st,
                 "Derivadas nulas en la submuestra: esos parámetros no aprenden" if st != "ok" else ""))
    ch.append(_c("|O_re| / |O_im|", f"{_fmt(m['o_re'])} / {_fmt(m['o_im'])}", "info",
                 "Sensibilidad media del módulo (Re) y de la fase (Im) de ψ"))
    rf = m["ntk_rank_frac"]
    st = "warn" if math.isfinite(rf) and rf < TH["ntk_rank_frac"] else "ok"
    ch.append(_c("NTK: autovalores > diag_shift",
                 f"{m['ntk_eff_rank']} de {2 * m['n_jacobian']}  (λ_max {_fmt(m['ntk_max'])})", st,
                 "diag_shift domina casi todo el espectro: SR ≈ descenso de gradiente lento" if st == "warn" else
                 "Direcciones que SR resuelve sin que domine la regularización"))
    c = m["ntk_cond"]
    st = "warn" if math.isfinite(c) and c > TH["ntk_cond"] else "ok"
    ch.append(_c("λ_max / diag_shift", _fmt(c, "{:.1e}"), st,
                 "Sistema muy mal condicionado" if st == "warn" else ""))
    for x in m["modules"]:
        if x.get("dead", 0) > TH["dead_warn"]:
            ch.append(_c(f"Módulo {x['label']}", f"muertos {100 * x['dead']:.0f} %", "warn",
                         "Módulo con muchas derivadas nulas"))
    return {"checks": ch, "modules": m["modules"]}


DIAGNOSE = {
    "energy": diag_energy,
    "sampling": diag_sampling,
    "optimization": diag_optimization,
    "gradients": diag_gradients,
    "wavefunction": diag_wavefunction,
    "local_energy": diag_local_energy,
    "magnetization": diag_magnetization,
}


def diagnose_metrics(metrics):
    diagnosis = {name: DIAGNOSE[name](m) for name, m in metrics.items()}
    worst = "ok"
    for d in diagnosis.values():
        for c in d["checks"]:
            if c["status"] in ("warn", "error") and RANK[c["status"]] > RANK[worst]:
                worst = c["status"]
    diagnosis["overall"] = {
        "status": worst,
        "message": {"ok": "Simulación saludable", "warn": "Revisar los avisos",
                    "error": "Problema serio: intervenir"}[worst],
    }
    return diagnosis
