import netket as nk


class MultipleRules(nk.sampler.rules.MultipleRules):

    def random_state(
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
    ):

        return self.rules[0].random_state(
            sampler,
            machine,
            params,
            sampler_state,
            key,
        )
