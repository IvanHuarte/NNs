import jax
import jax.numpy as jnp
import netket as nk

from NN_module.utils import print_tree

from .utils import append_tree_display, display_header

# import traceback
# jax.config.update("jax_debug_nans", True)


class LeavesGradient:

    def __init__(self, setup):

        self.do_each = setup["do_each"]
        self.metrics = setup["metrics"]
        self.metrics_list = self.metrics.split("|")
        self.show_parameters = setup["show_parameters"]

        self.header = display_header(self.metrics_list, self.show_parameters)

    def __call__(self, step, log_data, driver):

        if step % self.do_each == 0:
            vstate = driver.state
            params = vstate.parameters
            apply_fun = vstate._apply_fun
            samples = vstate.samples

            samples = samples.reshape((-1, samples.shape[-1]))


            # PSI
            logpsi = vstate.log_value(samples)
            print(jnp.min(jnp.real(logpsi)))
            print(jnp.max(jnp.real(logpsi)))
            print(f"IsNan: {jnp.isnan(logpsi).any()}")

            # ELOC
            Eloc = vstate.local_estimators(driver._ham)

            print(jnp.isnan(Eloc).any())
            print(jnp.isinf(Eloc).any())

            gradient = nk.jax.jacobian(apply_fun, params, samples, mode="complex")

            display_tree = jax.tree_util.tree_map(lambda x: "", params)

            print(self.header, "\n")

            # PARAMETERS
            if self.show_parameters:
                display_tree = append_tree_display(
                    display_tree, params, self.metrics_list, gradients=False
                )
                display_tree = jax.tree_util.tree_map(
                    lambda x: x + "    ||    ", display_tree
                )

            # GRADIENTS
            display_tree = append_tree_display(
                display_tree, gradient, self.metrics_list
            )

            print_tree(display_tree, values=True)
            print("\n")

        return True
