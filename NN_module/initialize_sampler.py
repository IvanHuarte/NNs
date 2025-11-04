from NN_module.samplers import RULES
from netket.sampler.rules import MultipleRules
from netket.sampler import MetropolisSampler


class SamplerFactory():

    
    def __init__(self, setup):
        self.setup = setup
        self.rules = [RULES[rule_name]() for rule_name in setup["sampler"]["rules"]]
        self.prules = setup["sampler"]["prules"] if len(self.rules) > 1 else [1.0]

        assert abs(sum(self.prules) - 1.0) < 1e-8, "The probabilities of the rules must sum to 1."

    def get_sampler(self, hilbert):
        
        rules_wrapper = MultipleRules(self.rules, self.prules) if len(self.rules) > 1 else self.rules[0]

        sampler = MetropolisSampler(
            hilbert=hilbert,
            rules=rules_wrapper,
            n_chains_per_rank=self.setup["n_chains_per_rank"],
            chunk_size=self.setup["chunk_sampler"],
        )

        return sampler