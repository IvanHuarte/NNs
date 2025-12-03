import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import os

os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "1"


### Functions used in self.gradient_metrics ###
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


def diagnose_metrics(self, metrics):
    """
    Analiza métricas y devuelve diagnóstico automático.

    Args:
        metrics: Diccionario de métricas de gradient_metrics, samples_autocorrelation, phase_stats

    Returns:
        dict: Diagnóstico por categoría con status y mensaje
    """
    diagnosis = {}

    # 1. DIAGNOSIS GRADIENTS
    grads = metrics["gradients"]

    # Normas principales
    norm_mean = grads["norm_per_sample"]["mean"]
    norm_std = grads["norm_per_sample"]["std"]
    global_norm = grads["global_norm"]

    # Status normas
    if norm_mean < 1e-3:
        norm_status = "💚💚💚 CONVERGIDO"
        norm_msg = "Gradientes muy pequeños → Buena convergencia"
    elif norm_mean < 1:
        norm_status = "✅ SANO"
        norm_msg = "Gradientes normales → Entrenando bien"
    elif norm_mean < 10:
        norm_status = "⚠️ ALTO"
        norm_msg = "Gradientes grandes → Monitorear LR 📊"
    else:
        norm_status = "🔴 EXPLOSIVO"
        norm_msg = "Gradientes peligrosos → Reduce LR! 📉"

    # Re/Im ratios
    reim_global = (
        grads["ReIm_norms"]["global"]["Re"] / grads["ReIm_norms"]["global"]["Im"]
    )
    mod_ratio = (
        grads["ReIm_norms"]["ModulusNet"]["Re"]
        / grads["ReIm_norms"]["ModulusNet"]["Im"]
    )
    phase_ratio = (
        grads["ReIm_norms"]["PhaseNet"]["Im"] / grads["ReIm_norms"]["PhaseNet"]["Re"]
    )

    reim_status = "🟢" if 0.5 < reim_global < 5 else "🟡"
    arch_status = (
        "🟢"
        if mod_ratio > 10 and phase_ratio > 10
        else "🟡" if mod_ratio > 1 and phase_ratio > 1 else "🔴"
    )

    # Análisis distribución gradientes
    intervals = grads["percentage_norm_intervals"]["log_intervals"]
    percentages = grads["percentage_norm_intervals"]["percentages"]

    grad_dist_status = "🟢"
    grad_dist_msg = []
    if percentages[0] > 20:  # >20% en <1e-4
        grad_dist_msg.append("Muchas normas nulas → Posible saturación")
        grad_dist_status = "🟡"
    if percentages[-1] > 5:  # >5% en >1e3
        grad_dist_msg.append("Gradientes explosivos detectados")
        grad_dist_status = "🔴"
    if sum(percentages[4:6]) > 70:  # >70% en 1-100
        grad_dist_msg.append("Entrenando activamente")

    diagnosis["gradients"] = {
        "status": f"{norm_status} | ReIm:{reim_status} | Arch:{arch_status} | Dist:{grad_dist_status}",
        "norm_per_sample": f"{norm_mean:.2e} ± {norm_std:.2e}",
        "global_norm": f"{global_norm:.2e}",
        "message": f"{norm_msg}. Re/Im global: {reim_global:.1f}. Arquitectura OK: {mod_ratio:.0f}/{phase_ratio:.0f}",
        "distribution": {"status": grad_dist_status, "alerts": grad_dist_msg},
    }

    # 2. DIAGNOSIS SAMPLES (MCMC)
    samples = metrics["samples"]
    acc_rate = samples["acceptance"]
    tau_corr = samples["tau_corr"]
    ess_eff = samples["efficiency"]

    mcmc_status = "🟢"
    mcmc_alerts = []

    if acc_rate < 0.3:
        mcmc_status = "🟡"
        mcmc_alerts.append("Acceptance baja → Aumenta step_size 📈")
    elif acc_rate > 0.8:
        mcmc_status = "🟡"
        mcmc_alerts.append("Acceptance alta → Reduce step_size 📉")

    if tau_corr > 20:
        mcmc_status = "🔴"
        mcmc_alerts.append("Alta autocorrelación → Sampler lento")
    elif tau_corr > 10:
        mcmc_status = "🟡"
        mcmc_alerts.append("Autocorrelación moderada")

    if ess_eff < 0.1:
        mcmc_status = "🟡"
        mcmc_alerts.append("Eficiencia baja")

    diagnosis["samples"] = {
        "status": mcmc_status,
        "acceptance": f"{acc_rate:.1%}",
        "tau_corr": f"{tau_corr:.1f}",
        "ESS_eff": f"{ess_eff:.1%}",
        "message": f"Sampler {'Excelente' if ess_eff>0.5 else 'Bueno' if ess_eff>0.2 else 'Mejorable'}. Alerts: {', '.join(mcmc_alerts)}",
    }

    # 3. DIAGNOSIS PHASE (Marshall)
    phase = metrics["phase"]["phase"]
    mean_phase = np.abs(phase["mean"])
    std_phase = phase["std"]

    phase_status = "🟢"
    phase_msg = "Fase Marshall OK"

    if mean_phase > 0.1:
        phase_status = "🟡"
        phase_msg = "Fase con sesgo → Revisa Marshall"
    if std_phase > 2:
        phase_status = "🟡"
        phase_msg += " | Alta varianza de fase"

    diagnosis["phase"] = {
        "status": phase_status,
        "mean_phase": f"{mean_phase:.2f}",
        "std_phase": f"{std_phase:.2f}",
        "message": phase_msg,
    }

    # 4. STATUS GENERAL
    statuses = [d["status"][0] for d in diagnosis.values()]
    overall_status = (
        "🟢"
        if all(s == "🟢" for s in statuses)
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


class SanityMonitor:

    def __init__(self):
        pass

    def gradient_metrics(self, params, apply_fun, s_batch):
        """Compute relevant metrics of the gradients.

        Args:
            params: The parameters of the model.
            apply_fun: The apply function of the model.
            s_batch: A batch of samples.

        Returns:
            A dictionary containing the mean and standard deviation of the
            gradients' absolute values.
        """

        grad_metrics = {}

        O = nk.jax.jacobian(apply_fun, params, s_batch, mode="complex")

        # Mean Norm per sample ⟨||∇logψ(s)||⟩ (axis=1 para (Nsamples,...))
        grad_norms_per_sample = tree_norm(O, axis=1)
        mean_norm_sample = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, grad_norms_per_sample)
        ) / len(s_batch)
        total_norm_std = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.std, grad_norms_per_sample)
        )
        grad_metrics["norm_per_sample"] = {
            "mean": mean_norm_sample,
            "std": total_norm_std,
        }

        ### Global Norm ||∇logψ|| normalized
        global_norm = tree_global_norm(O) / jnp.sqrt(len(s_batch))
        grad_metrics["global_norm"] = global_norm

        # Real/Imaginary components stats
        # In global norm
        O_re, O_im = extract_real_imag(O)
        O_real_norm = tree_global_norm(O_re)
        O_imag_norm = tree_global_norm(O_im)

        # In Modulus and Phase nets
        O_modulus = O["ModulusNet"]
        O_phase = O["PhaseNet"]
        O_mod_re, O_mod_im = extract_real_imag(O_modulus)
        O_ph_re, O_ph_im = extract_real_imag(O_phase)
        norms_mod_re = tree_norm(O_mod_re, axis=0)  # Normas por muestra
        norms_mod_im = tree_norm(O_mod_im, axis=0)
        norms_ph_re = tree_norm(O_ph_re, axis=0)
        norms_ph_im = tree_norm(O_ph_im, axis=0)
        O_mod_re_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_mod_re)
        )
        O_mod_im_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_mod_im)
        )
        O_ph_re_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_ph_re)
        )
        O_ph_im_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(jnp.mean, norms_ph_im)
        )

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

        # Percentage of leaves in different norm intervals
        threshold_points = jnp.logspace(-4, 3, num=8)
        per_above = list(
            map(
                lambda threshold: percentage_above_threshold(
                    grad_norms_per_sample, threshold
                ),
                threshold_points,
            )
        )
        per_below = [100 - per_above[0]]

        percentaje_per_norm = per_below + [
            per_above[i] - per_above[i + 1] for i in range(len(per_above) - 1)
        ]

        # PRINTS
        print(f"⟨||∇logψ||⟩ = {mean_norm_sample} ± {total_norm_std:.2e}")
        print(f"||∇global|| = {global_norm:.2e}")
        print(f"Re/Im ratio: {O_real_norm/O_imag_norm:.2f}")
        print(f"<1e-4: {100-per_above[0]:.3f}%\n")
        for i in range(len(threshold_points)):
            if i != len(threshold_points) - 1:
                print(
                    f"Interval {threshold_points[i]:.0e} <---> {threshold_points[i+1]:.0e}: {percentaje_per_norm[i+1]:.3f}%"
                )
                print()

        grad_metrics["percentage_norm_intervals"] = {
            "log_intervals": threshold_points,
            "percentages": percentaje_per_norm,
        }

        return grad_metrics

    def samples_autocorrelation(self, driver, vstate, samples):
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

        # Local energy (energy of each sample)
        Eloc = vstate.local_estimators(driver._ham)
        Eloc_flat = Eloc.flatten()
        n_samples = Eloc_flat.shape[0]

        stats = nk.stats.statistics(Eloc_flat)

        if hasattr(stats, "tau_corr_max"):
            tau_corr = (
                stats.tau_corr_max
                if not np.isnan(stats.tau_corr_max)
                else stats.tau_corr
            )
        else:
            tau_corr = stats.tau_corr

        ESS = n_samples / (1 + 2 * tau_corr)
        efficiency = ESS / n_samples

        print("=== SAMPLES AUTOCORRELATION ===")
        print(f"Samples: {samples.shape}")
        print(f"τ_corr = {stats.tau_corr:.2f} ")
        print(f"ESS (Neff) = {ESS:.0f}/{n_samples} ({efficiency:.2f})")

        acceptance = driver.state.sampler_state.acceptance

        acorr_metrics = {
            "acceptance": acceptance,
            "tau_corr": tau_corr,
            "ESS": ESS,
            "efficiency": efficiency,
        }
        return acorr_metrics

    def phase_stats(self, params, apply_fun, s_batch):
        """Compute relevant metrics of the phase of the wavefunction.

        Args:
            params: The parameters of the model.
            apply_fun: The apply function of the model.
            s_batch: A batch of samples.

        Returns:
            A dictionary containing the mean and standard deviation of the
            phase values.
        """

        phase_metrics = {}

        log_psi = apply_fun(params, s_batch)
        phases = np.imag(log_psi)
        exp_phases = jnp.exp(1j * phases)

        mean_phase = jnp.mean(exp_phases)
        std_phase = jnp.std(exp_phases)

        phase_metrics["phase"] = {
            "mean": mean_phase,
            "std": std_phase,
        }

        # PRINTS
        print(f"phase = {mean_phase:.2f} ± {std_phase:.2f}")

        return phase_metrics

    def __call__(self, step, log_data, driver):
        """Monitor the sanity of the training by printing relevant metrics.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        vstate = driver.state
        params = vstate.parameters
        apply_fun = vstate._apply_fun
        s_batch = vstate.samples.reshape(-1, vstate.hilbert.size)

        energy_step = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        vscore_step = self.N * var / mean**2

        grad_metrics = self.compute_grad_metrics(params, apply_fun, s_batch)

        print(
            f"Step {step}: Energy = {energy_step:.6f}, Variance = {var:.6f}, V-score = {vscore_step:.6f}"
        )

        return True
