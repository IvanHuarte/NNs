from NN_module.samplers import RULES
import jax.numpy as jnp
from netket.sampler.rules import MultipleRules
from netket.sampler import MetropolisSampler


class SamplerFactory:

    def __init__(self, setup, **kwargs):
        self.setup = setup
        self.kwargs = kwargs

    def _init_rules(self):

        self.rules = []
        for rule_name in self.setup["sampler"]["rules"]:

            raw_rule = RULES[rule_name]

            if rule_name == "ExchangeJ1J2":
                cm_model = self.kwargs["cm_model"]
                lat = cm_model.cm.lattice

                J1_nn = []
                J2_nn = []
                for neighs in lat.neighbors.values():
                    J1_nn.append(neighs[0])
                    J2_nn.append(neighs[1])

                try:
                    J1_nn = jnp.array(J1_nn)
                    J2_nn = jnp.array(J2_nn)
                except:
                    raise ValueError(
                        f"This lattice cannot be implemented fro J1-J2 exchange rule, since have irregular neighbors array."
                    )

                init_rule = raw_rule(
                    J1=cm_model.J1,
                    J2=cm_model.J2,
                    J1_nn=J1_nn,
                    J2_nn=J2_nn,
                )

            else:
                init_rule = raw_rule()

            self.rules.append(init_rule)

        self.prules = self.setup["sampler"]["prules"] if len(self.rules) > 1 else [1.0]

        rules_wrapper = (
            MultipleRules(self.rules, self.prules)
            if len(self.rules) > 1
            else self.rules[0]
        )
        assert (
            abs(sum(self.prules) - 1.0) < 1e-8
        ), "The probabilities of the rules must sum to 1."

        return rules_wrapper

    def get_sampler(self, hilbert):

        rules_wrapper = self._init_rules()

        sampler = MetropolisSampler(
            hilbert=hilbert,
            rule=rules_wrapper,
            n_chains_per_rank=self.setup["n_chains_per_rank"],
            chunk_size=self.setup["chunk_sampler"],
        )

        return sampler
