def msg_reim_ratio(ratio, name):
    if name == "global":
        if 0.5 < ratio < 5:
            return "Razón Re/Im equilibrada en el global."
        else:
            return "Desequilibrio global Re/Im → revisar estabilidad del entrenamiento."

    if name == "mod":
        if ratio > 10:
            return "Gradientes del módulo dominantes y estables."
        elif ratio > 1:
            return "Gradientes del módulo aceptables pero con cierto desequilibrio."
        else:
            return "Módulo demasiado dominado por Im → posible colapso del canal real."

    if name == "phase":
        if ratio > 10:
            return "Gradientes de la fase dominantes y estables."
        elif ratio > 1:
            return "Fase OK pero con desequilibrio moderado."
        else:
            return (
                "Fase dominada por Re → posible inestabilidad en la parte imaginaria."
            )


def diagnose_gradients(metrics):

    diagnosis = {}

    # 1. DIAGNOSIS GRADIENTS
    grads = metrics["gradients"]

    ### 1.- Norm per sample diagnosis
    norm_mean = grads["norm_per_sample"]["mean"]
    norm_std = grads["norm_per_sample"]["std"]

    # Mean
    label_mean = f"⟨||∇logψ||⟩_s = {norm_mean:.2e}"
    if norm_mean < 1e-3:
        norm_status = "💚💚💚"
        norm_msg = "CONVERGIDO. Gradientes muy pequeños → Buena convergencia"
    elif norm_mean < 1:
        norm_status = "✅"
        norm_msg = "SANO. Gradientes normales → Entrenando bien"
    elif norm_mean < 10:
        norm_status = "⚠️"
        norm_msg = "ALTO. Gradientes grandes → Monitorear LR 📊"
    elif norm_mean < 100:
        norm_status = "⚠️⚠️"
        norm_msg = " MUY ALTO. Gradientes muy grandes → Reducir un poco LR 📊"
    elif norm_mean > 100:
        norm_status = "🔴"
        norm_msg = " EXPLOSIVO. Gradientes peligrosos → Reduce LR! 📉"
    else:
        norm_status = "❓"
        norm_msg = " DESCONOCIDO. Valor de gradientes inesperado → Revisa"

    # Std
    std_ratio = norm_std / norm_mean
    label_std = f"σ[||∇logψ||]_s = {norm_std:.2e}  (σ/μ={std_ratio:.2f})"
    if std_ratio < 1:
        std_status = "🟢"
        std_msg = "Variabilidad normal entre muestras."
    elif std_ratio < 2:
        std_status = "🟡"
        std_msg = "Variabilidad moderada, posible sampling disbalanceado."
    elif std_ratio < 5:
        std_status = "⚠️"
        std_msg = "Alta variabilidad entre muestras, revisar sampler."
    else:
        std_status = "🔴"
        std_msg = "Muy alta variabilidad, sampling problemático!"

    diagnosis["norm_per_sample"] = {
        "mean": {"label": label_mean, "status": norm_status, "message": norm_msg},
        "std": {"label": label_std, "status": std_status, "message": std_msg},
    }

    # --- Global norm ---
    global_norm = grads["global_norm"]
    label_global = f"||∇logψ||_global = {global_norm:.2e}"

    if global_norm < 1:
        global_status = "🟢"
        global_msg = "Escala de gradientes normal."
    elif global_norm < 10:
        global_status = "🟡"
        global_msg = "Escala moderada, monitorear entrenamiento."
    elif global_norm < 100:
        global_status = "⚠️"
        global_msg = "Gradientes grandes, posible inestabilidad."
    elif global_norm >= 100:
        global_status = "🔴"
        global_msg = "Gradientes explosivos, acción urgente!"
    else:
        global_status = "❓ DESCONOCIDO"
        global_msg = "Valor de gradientes inesperado → Revisa"

    diagnosis["global_norm"] = {
        "label": label_global,
        "status": global_status,
        "message": global_msg,
    }

    # Re/Im ratios
    reim_global_re = grads["ReIm_norms"]["global"]["Re"]
    reim_global_im = grads["ReIm_norms"]["global"]["Im"]
    reim_mod_re = grads["ReIm_norms"]["ModulusNet"]["Re"]
    reim_mod_im = grads["ReIm_norms"]["ModulusNet"]["Im"]
    reim_phase_re = grads["ReIm_norms"]["PhaseNet"]["Re"]
    reim_phase_im = grads["ReIm_norms"]["PhaseNet"]["Im"]

    global_ratio = reim_global_re / reim_global_im
    mod_ratio = reim_mod_re / reim_mod_im
    phase_ratio = reim_phase_im / reim_phase_re

    label_global_ratio = f"||Re(∇ψ)|| / ||Im(∇ψ)|| = {reim_global_re:.2e} / {reim_global_im:.2e} = {global_ratio:.2f}"
    label_mod_ratio = f"||∇ψ_ph||_Re / ||∇ψ_ph||_Im = {reim_mod_re:.2e} / {reim_mod_im:.2e} = {mod_ratio:.2f}"
    label_phase_ratio = f"||∇ψ_ph||_Im / ||∇ψ_ph||_Re = {reim_phase_im:.2e} / {reim_phase_re:.2e} = {phase_ratio:.2f}"

    global_status = "🟢" if 0.5 < global_ratio < 5 else "🟡"
    mssg_global = msg_reim_ratio(global_ratio, "global")

    mod_status = "🟢" if mod_ratio > 10 else "🟡" if mod_ratio > 1 else "🔴"
    mssg_mod = msg_reim_ratio(mod_ratio, "mod")

    phase_status = "🟢" if phase_ratio > 10 else "🟡" if phase_ratio > 1 else "🔴"
    mssg_phase = msg_reim_ratio(phase_ratio, "phase")

    diagnosis["ReIm_norms"] = {
        "global": {
            "label": label_global_ratio,
            "status": global_status,
            "message": mssg_global,
        },
        "ModulusNet": {
            "label": label_mod_ratio,
            "status": mod_status,
            "message": mssg_mod,
        },
        "PhaseNet": {
            "label": label_phase_ratio,
            "status": phase_status,
            "message": mssg_phase,
        },
    }

    # Análisis distribución gradientes
    threshold_points = grads["percentage_norm_intervals"]["log_intervals"]
    percentages = grads["percentage_norm_intervals"]["percentages"]

    label_percentages = [f"    < 1e-04:        {percentages[0]:.3f} %"]
    for i in range(len(threshold_points)):
        if i != len(threshold_points) - 1:
            label_percentages.append(
                f"{threshold_points[i]:.0e} <--> {threshold_points[i+1]:.0e}:   {percentages[i+1]:.3f} %"
            )
    label_percentages.append(f"   > 1e+03:        {percentages[-1]:.3f} %")

    label_percentages = label_percentages[::-1]

    grad_dist_status = "🟢"
    grad_dist_msg = []
    if percentages[0] > 20:  # >20% en <1e-4
        grad_dist_msg = "Muchas normas nulas → Posible saturación"
        grad_dist_status = "🟡"
    if percentages[-1] > 5:  # >5% en >1e3
        grad_dist_msg = "Gradientes explosivos detectados"
        grad_dist_status = "🔴"
    if sum(percentages[4:6]) > 70:  # >70% en 1-100
        grad_dist_msg = "Entrenando activamente"

    diagnosis["percentage_norm_intervals"] = {
        "label": label_percentages,
        "status": grad_dist_status,
        "message": grad_dist_msg,
    }

    return diagnosis


def diagnose_sampling(metrics):

    diagnosis = {}

    samples = metrics["sampling"]

    acc_rate = samples["acceptance"]
    tau_corr = samples["tau_corr"]
    ess = samples["ESS"]
    ess_eff = samples["efficiency"]
    n_samples = samples["n_samples"]

    label_accept = f"Acc = {acc_rate:.3f}"
    label_tau_corr = f"τ_corr = {tau_corr:.2f}"
    label_ess = f"ESS / N (eff)= {ess:.0f} / {n_samples} ({ess_eff:.2f})"

    if acc_rate < 0.3:
        acc_status = "🟡"
        acc_msg = "Acceptance baja → Aumenta step_size 📈"
    elif acc_rate > 0.75:
        acc_status = "🟡"
        acc_msg = "Acceptance alta → Reduce step_size 📉"
    else:
        acc_status = "🟢"
        acc_msg = "Acceptance OK"

    if tau_corr > 20:
        tau_status = "🔴"
        tau_msg = "Alta autocorrelación → Sampler lento"
    elif tau_corr > 10:
        tau_status = "🟡"
        tau_msg = "Autocorrelación moderada"
    else:
        tau_status = "🟢"
        tau_msg = "Autocorrelación baja"

    if ess_eff < 0.1:
        esseff_status = "🟡"
        esseff_msg = "Eficiencia baja"
    else:
        esseff_status = "🟢"
        esseff_msg = "Eficiencia aceptable"

    diagnosis = {
        "acceptance": {"label": label_accept, "status": acc_status, "message": acc_msg},
        "tau_corr": {"label": label_tau_corr, "status": tau_status, "message": tau_msg},
        "ESS": {"label": label_ess, "status": esseff_status, "message": esseff_msg},
    }

    return diagnosis


def diagnose_sample_histogram(histogram):

    diagnosis = {}

    magnetization, counts = (
        histogram["sample_histogram"]["magnetization"],
        histogram["sample_histogram"]["counts"],
    )

    label_hist = {
        "magnetization": [f"M = {m}" for m in magnetization],
        "counts": [f"{c}" for c in counts],
    }

    diagnosis["histogram"] = {
        "label": label_hist,
        "status": "🟢",
        "message": "Histograma de magnetización OK",
    }

    return diagnosis


def diagnose_phase(metrics):

    diagnosis = {}

    phase_main = metrics["phase"]

    for q, (key, value) in enumerate(phase_main.items()):

        mean_phase_abs = value["mean"]
        std_phase = value["std"]

        label_phase_mean = f"{mean_phase_abs:.3f}"
        label_phase_std = f"{std_phase:.3f}"

        # 3. DIAGNOSIS PHASE

        if mean_phase_abs > 0.8:
            phase_mean_status = "🟢"
            phase_mean_msg = "Estructura detectada"
        elif mean_phase_abs > 0.3:
            phase_mean_status = "🟡"
            phase_mean_msg = "Moderado. ¿Esta en combinacion con otros?"
        else:
            phase_mean_status = "🔴"
            phase_mean_msg = "Armonico nulo"

        if std_phase > 3:
            phase_std_status = "🟡"
            phase_std_msg = "Alta varianza de fase"
        else:
            phase_std_status = "🟢"
            phase_std_msg = "Varianza de fase aceptable"

        diagnosis[key] = {
            "mean": {
                "label": label_phase_mean,
                "status": phase_mean_status,
                "message": phase_mean_msg,
            },
            "std": {
                "label": label_phase_std,
                "status": phase_std_status,
                "message": phase_std_msg,
            },
        }
    return diagnosis


diagnose_dict = {
    "gradients": diagnose_gradients,
    "sampling": diagnose_sampling,
    "sample_histogram": diagnose_sample_histogram,
    "phase": diagnose_phase,
}


def diagnose_metrics(metrics):
    """
    Analiza métricas y devuelve diagnóstico automático.

    Args:
        metrics: Diccionario de métricas de gradient_metrics, samples_autocorrelation, phase_stats

    Returns:
        dict: Diagnóstico por categoría con status y mensaje
    """

    diagnosis = {}

    for metric_type in metrics.keys():
        if metric_type in diagnose_dict:
            diag_func = diagnose_dict[metric_type]
            diag_result = diag_func(metrics)
            diagnosis[metric_type] = diag_result
        else:
            NotImplementedError(
                f"Diagnosis for metric '{metric_type}' not implemented."
            )

    # 4. GENERAL STATUS
    def recursive_extract_status(d):
        """Extrae recursivamente los estados de un diccionario anidado."""
        statuses = []
        for key, value in d.items():
            if isinstance(value, dict):
                statuses.extend(recursive_extract_status(value))
            elif key == "status":
                statuses.append(d[key])
        return statuses

    statuses = recursive_extract_status(diagnosis)
    overall_status = (
        "🟢"
        if all(s in ["🟢", "✅", "💚💚💚"] for s in statuses)
        else "🟡" if "🔴" not in statuses else "🔴"
    )

    diagnosis["overall"] = {
        "status": overall_status,
        "message": (
            "Simulación saludable"
            if overall_status == "🟢"
            else "Monitorear" if overall_status == "🟡" else "Intervenir"
        ),
    }

    return diagnosis
