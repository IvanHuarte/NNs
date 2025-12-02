import numpy as np
import jax
import jax.numpy as jnp
import netket as nk


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

        # % Explosivas
        grad_norms_sample = tree_norm(O, axis=1)
        explosive_pct = (
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda x: jnp.mean(x > 1e3), grad_norms_sample),
            )
            * 100
        )
        grad_metrics["explosive_pct"] = explosive_pct

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
