import jax.numpy as jnp
from netket.sampler import ARDirectSampler, MetropolisSampler

from Hydra.sampler import RULES
from Hydra.sampler.custom_rules import MultipleRules


def get_neighbor_array(neighbors_dict):
    N = len(neighbors_dict)
    nn_max = len(neighbors_dict[0])
    nn_dim = [len(neighbor) for neighbor in neighbors_dict[0]]
    max_per_shell = max(nn_dim)

    neighbors_array = -jnp.ones((N, nn_max, max_per_shell), dtype=jnp.int64)

    for i in range(N):
        for s in range(nn_max):
            neigh = neighbors_dict[i][s]
            neighbors_array = neighbors_array.at[i, s, : len(neigh)].set(
                jnp.array(neigh)
            )
    return neighbors_array, nn_max, nn_dim


def resolve_sampler_setup(setup, verbose=False):
    """
    General sampler settings (n_chains_per_rank, n_samples_per_chain,
    chunk_sampler, chunk_vstate, ...) overridden by the ones defined in the
    block of the selected sampler, e.g. setup["metropolis"]["n_chains_per_rank"].
    Idempotent: resolving an already resolved setup changes nothing.
    """
    selection = setup["selection"]
    block = setup.get(selection, {}) or {}

    resolved = dict(setup)
    for k, v in block.items():
        if isinstance(v, dict):  # e.g. metropolis["sampler"] (rules)
            continue
        if verbose and k in setup and setup[k] != v:
            print(f"[Sampler] '{selection}' sets {k} = {v} (general value {setup[k]} ignored)")
        resolved[k] = v

    return resolved


class SamplerFactory:

    def __init__(self, setup, **kwargs):
        self.setup = resolve_sampler_setup(setup)
        self.kwargs = kwargs

    def _init_rules(self):

        self.rules = []
        for rule_name in self.setup["metropolis"]["sampler"]["rules"]:

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
                        "This lattice cannot be implemented fro J1-J2 exchange rule, since have irregular neighbors array."
                    )

                init_rule = raw_rule(
                    J1=cm_model.J1,
                    J2=cm_model.J2,
                    J1_nn=J1_nn,
                    J2_nn=J2_nn,
                )
            elif rule_name == "Exchange":
                cm_model = self.kwargs["cm_model"]
                neighbors_dict = cm_model.cm.lattice.neighbors
                neighbors_array, nn_max, nn_dim = get_neighbor_array(neighbors_dict)
                init_rule = raw_rule(
                    neighbors=neighbors_array,
                    nn_max=int(nn_max),
                    nn_dim=jnp.array(nn_dim, dtype=jnp.int32),
                )

            else:
                init_rule = raw_rule()

            self.rules.append(init_rule)

        rules_setup = self.setup["metropolis"]["sampler"]
        rule_names = rules_setup["rules"]

        # With a single rule its probability is 1 ('prules' is ignored)
        if len(self.rules) > 1:
            self.prules = rules_setup["prules"]
            if len(self.prules) != len(self.rules):
                raise ValueError(
                    f"sampler.metropolis.sampler: {len(rule_names)} rules {rule_names} "
                    f"but {len(self.prules)} probabilities prules={self.prules}."
                )
        else:
            self.prules = [1.0]

        if abs(sum(self.prules) - 1.0) > 1e-8:
            raise ValueError(
                f"sampler.metropolis.sampler.prules={self.prules} must sum to 1."
            )

        # initial_state: "hilbert", "M0", "Mmax" or null (rule's own default).
        # MultipleRules applies it, also when there is a single rule.
        initial_state = rules_setup.get("initial_state")
        if len(self.rules) > 1 or initial_state is not None:
            rules_wrapper = MultipleRules(self.rules, self.prules, initial_state)
        else:
            rules_wrapper = self.rules[0]

        return rules_wrapper

    def get_sampler(self, hilbert):

        selection = self.setup["selection"]

        if selection == "metropolis":
            rules_wrapper = self._init_rules()
            sampler = MetropolisSampler(
                hilbert=hilbert,
                rule=rules_wrapper,
                n_chains_per_rank=self.setup["n_chains_per_rank"],
                chunk_size=self.setup["chunk_sampler"],
            )

        elif selection == "nkARSampler":
            sampler = ARDirectSampler(hilbert=hilbert)

        else:
            raise NotImplementedError(f"Sampler {selection} not implemented yet")

        return sampler
