import copy
import glob
import os

import jax
import jax.numpy as jnp
from netket.sampler.metropolis import MetropolisSamplerState

from Hydra.NN.NeuralNetwork import NeuralNetwork
from Hydra.NN.utils import (
    WHOLE_NETWORK_CODE,
    change_module_attr,
    flatten_with_paths,
    get_code2path_flatten,
    get_code2path_tree,
    get_subtree,
    greedy_transplant,
    load_from_file,
    print_architecture,
    set_subtree,
    setup_from_template,
    upgrade_legacy_params,
)


def resolve_load_path(path):
    """
    load_from entries may be a results json/orbax path (as before), a
    directory or a glob pattern: then the most recent '*results*.json' found
    (recursively) is used. Lets chained jobs (e.g. excited states started
    from a ground state still to be computed) point to a write folder.
    """
    if not (os.path.isdir(path) or any(c in path for c in "*?[")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"load_from: {path} does not exist on this machine")
        return path
    if path.rstrip("/").endswith("_best.orbax") or "_best.orbax/" in path:
        return path  # explicit on-disk best (BestIterKeeper)
    pattern = os.path.join(path, "**", "*results*.json") if os.path.isdir(path) else path
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not matches and os.path.isdir(path):
        # The source run did not finish (killed, crashed, still running):
        # fall back to the bests BestIterKeeper wrote along the run.
        bests = sorted(
            glob.glob(os.path.join(path, "**", "*_best.orbax"), recursive=True),
            key=os.path.getmtime,
        )
        if bests:
            print(f"load_from: {path} has no results json, using {bests[-1]}")
            return bests[-1]
    if not matches:
        raise FileNotFoundError(f"load_from: no results json matches {pattern}")
    print(f"load_from: {path} -> {matches[-1]}")
    return matches[-1]


def _module_names(setup):
    """Names of all modules appearing in a setup tree."""
    names = set()
    if isinstance(setup, dict):
        if "module" in setup:
            names.add(setup["module"])
        for v in setup.values():
            names |= _module_names(v)
    return names


class Hydra(NeuralNetwork):
    """
    Dynamic neural architecture manager with evolutionary stages.

    Hydra extends NeuralNetwork by allowing the architecture to evolve
    over multiple stages, where each stage may:
        - Change the network topology.
        - Modify module attributes.
        - Update symmetry constraints.
        - Transplant weights from previous models.

    The object persists across stages while the underlying model
    is reconstructed in-place.
    """

    def __init__(self, hydra_config, **external_kwargs):
        """
        Initialize Hydra with an evolutionary configuration.

        Parameters
        ----------
        hydra_config : dict
            Global Hydra configuration dictionary. Must define:
                - architecture templates
                - evolution stages
                - storage and symmetry rules

        **external_kwargs : dict
            External parameters required by the models
            (e.g. lattice_size, physical constants).
        """

        self.n_stage = 0

        self.print_arch = hydra_config["print_arch"]
        self.load_model = hydra_config["load_model"]

        # Copies: Hydra modifies these along the stages and must not
        # alter the user configuration (which is also saved to disk).
        self.storage = copy.deepcopy(hydra_config["storage"])
        self.symm_wrapper = copy.deepcopy(hydra_config["symm_wrapper"])
        self.evolution_name = hydra_config["selection"]
        self.arch_evolution = copy.deepcopy(hydra_config[self.evolution_name])
        self.external_args = external_kwargs

        # Stage 0
        self.template = self.arch_evolution["stage0"]["template"]
        self.save_params = self.arch_evolution["stage0"].get("save_params", False)

        # Build the initial NN model. Adapt first template 'model_0' to a configuration dictionary.
        setup = setup_from_template(self.template, self.storage, self.symm_wrapper)
        # print_tree(setup, values=True)
        self.save_stage_setup(setup)

        super().__init__(setup, **self.external_args)

        if "lattice_size" in self.external_args:
            self.update_info()

        # Load initial parameters for stage0, in the case. If a valid artifact is set,
        # load info will be required in stage0 and must refeer the label set in params_history
        # at below, stageX as default. Loaded parameters should be transplanted to stage0
        # parameters PyTree after generate the variational state object via weight transplantation.

        self.params_history = {}
        self.prev_params = None  # Previous stage params for default transplant
        if self.load_model["parameters"] or self.load_model["sampler"]:
            self.load = self.arch_evolution["stage0"]["load"]
        else:
            self.load_model = None

    def save_stage_setup(self, setup):
        # Copy before building: initialize_from_setup modifies it in place.
        self.storage[f"stage{self.n_stage}"] = copy.deepcopy(setup)

    def setup_from_template(self, template, storage, symm_wrappers):
        return setup_from_template(template, storage, symm_wrappers)

    def get_code2path(self, params):
        return get_code2path_flatten(params)

    def draw_architecture(self, params=None, show_attrs=False):
        """
        Box diagram of the current architecture with the path code of each
        module and the operation that merges each pair of branches.
        If params is None the model is initialized to obtain the codes.
        """
        from Hydra.NN.arch_diagram import draw_architecture

        if params is None:
            params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.N)))
        return draw_architecture(self.setup, params, show_attrs=show_attrs)

    @staticmethod
    def _code_to_path(c2p, code):
        """Path of a code; WHOLE_NETWORK_CODE ('A') is the root of the tree."""
        return () if code == WHOLE_NETWORK_CODE else c2p[code]

    def update_info(self, N=None):
        """
        Compute and display architecture metadata.

        Prints:
            - Hierarchical architecture tree.
            - Total number of parameters.
            - Total memory footprint.

        Parameters
        ----------
        N : int, optional
            Input dimension. If not provided, inferred from lattice_size.
        """

        assert (
            "lattice_size" in self.external_args or N != None
        ), "No size info (N) in internal state 'external_args' nor local "
        if N is None:
            self.N = int(jnp.prod(jnp.array(self.external_args["lattice_size"])))
        else:
            self.N = N

        params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.N)))

        print("********************* ARCHITECTURE INFO *********************\n")
        print(self.draw_architecture(params))
        print("\n")
        code2path_tree = get_code2path_tree(params["params"])
        print_architecture(code2path_tree, print_all=self.print_arch)
        print("\n")
        self.n_params, self.nbytes = self.get_params_info(self.model, self.N, True)
        print("\n*************************************************************")

    def arch_evol(self, old_params):
        """
        Advance to the next evolutionary stage.

        This method:
            - Saves current parameters if requested.
            - Loads the next stage configuration.
            - Updates symmetry settings.
            - Rebuilds the neural network architecture.
            - Updates internal metadata.

        Parameters
        ----------
        old_params : pytree
            Parameters of the previous model, used for later transplantation.
        """
        # PREVIOUS EON
        if self.save_params:
            print(f"Saving stage{self.n_stage} parameters....")
            self.params_history[f"stage{self.n_stage}"] = old_params
            print("Saved.")
        prev_stage = f"stage{self.n_stage}"
        self.prev_params = old_params

        # NEXT EON
        self.n_stage += 1

        stage_config = self.arch_evolution[f"stage{self.n_stage}"]
        self.template = stage_config["template"]
        self.save_params = stage_config.get("save_params", False)

        # Weight transplantation instructions. Without a "load" key, every
        # compatible weight of the previous stage is transplanted.
        # An explicit empty list ("load": []) means random initialization.
        if "load" in stage_config:
            self.load = stage_config["load"]
            if not self.load:
                print(
                    f"[Hydra] stage{self.n_stage}: 'load' is empty -> "
                    "random initialization, no weights transplanted."
                )
        else:
            self.load = [[prev_stage, WHOLE_NETWORK_CODE, WHOLE_NETWORK_CODE]]
            print(
                f"[Hydra] stage{self.n_stage}: no 'load' given -> transplanting "
                f"all compatible weights from {prev_stage}."
            )

        # Update symmetry settings, in the case.
        if "symm_wrapper" in stage_config:
            for symm_name, symm_setup in stage_config["symm_wrapper"].items():
                assert symm_name in self.symm_wrapper, (
                    f"Unknown symmetry wrapper '{symm_name}'. "
                    f"Available: {list(self.symm_wrapper)}"
                )
                for k, v in symm_setup.items():
                    assert k in self.symm_wrapper[symm_name], (
                        f"Unknown option '{k}' for symmetry wrapper '{symm_name}'. "
                        f"Available: {list(self.symm_wrapper[symm_name])}"
                    )
                    self.symm_wrapper[symm_name][k] = v

        # Create stage setup and change module attributes, in the case
        setup = setup_from_template(self.template, self.storage, self.symm_wrapper)
        if stage_config.get("change_attr"):
            present = _module_names(setup)
            for target in stage_config["change_attr"]:
                if target not in present:
                    print(
                        f"[Hydra] WARNING stage{self.n_stage}: change_attr target "
                        f"'{target}' not found in the architecture "
                        f"(modules: {sorted(present)}). Change ignored."
                    )
            setup, _ = change_module_attr(
                setup, copy.deepcopy(stage_config["change_attr"])
            )

        # Save the stage architecture setup
        self.save_stage_setup(setup)

        # Create new NN model
        self.initialize_from_setup(setup, self.external_args)

        # Show architecture and update metadata
        self.update_info(self.N)

    def weight_transplantation(self, new_params):
        """
        Transplant compatible weights from previous models.

        This method performs block-wise greedy transplantation between
        parameter trees based on precomputed anchor paths.

        For each load instruction:
            1. Extract source and target subtrees.
            2. Match compatible parameters using greedy strategy.
            3. Copy values into the new parameter tree.

        Parameters
        ----------
        new_params : pytree
            Parameters of the newly constructed model.

        code2path : bool, default=True
            If True, also return the updated code-to-path mapping.

        Returns
        -------
        new_params : pytree
            Updated parameter tree with transplanted weights.

        new_c2p : dict (optional)
            Mapping from codes to absolute parameter paths.
        """

        new_c2p = self.get_code2path(new_params)

        if self.load:
            print("\n******************* WEIGHT TRANSPLANTATION *******************")

        for params_name, old_code, new_code in self.load:

            if params_name in self.params_history:
                old_params = self.params_history[params_name]
            elif (
                params_name == f"stage{self.n_stage - 1}"
                and self.prev_params is not None
            ):
                old_params = self.prev_params
            else:
                raise KeyError(
                    f"[Hydra] Parameters '{params_name}' not available for "
                    f"transplantation. Saved: {list(self.params_history)}. "
                    "Set 'save_params': true in that stage."
                )
            old_c2p = self.get_code2path(old_params)

            for code, c2p, label in [
                (old_code, old_c2p, params_name),
                (new_code, new_c2p, f"stage{self.n_stage}"),
            ]:
                if code != WHOLE_NETWORK_CODE and code not in c2p:
                    valid = {k: "/".join(v) for k, v in c2p.items() if len(k) <= 2}
                    raise KeyError(
                        f"[Hydra] load {[params_name, old_code, new_code]}: code "
                        f"'{code}' does not exist in {label}. Valid codes: "
                        f"'{WHOLE_NETWORK_CODE}' (whole network) and, up to "
                        f"depth 2, {valid}"
                    )
            old_path_block = self._code_to_path(old_c2p, old_code)
            new_path_block = self._code_to_path(new_c2p, new_code)

            old_params_block = get_subtree(old_params, old_path_block)
            new_params_block = get_subtree(new_params, new_path_block)

            trasplant = greedy_transplant(old_params_block, new_params_block)

            n_old = len(flatten_with_paths(old_params_block))
            n_new = len(flatten_with_paths(new_params_block))
            moved = 0
            for old_rel, new_rel in trasplant:
                old_abs = old_path_block + old_rel
                new_abs = new_path_block + new_rel

                val = get_subtree(old_params, old_abs)
                target = get_subtree(new_params, new_abs)
                val = jnp.asarray(val, dtype=target.dtype)  # precision of the new network
                moved += val.size
                new_params = set_subtree(new_params, new_abs, val)

            print(
                f"{params_name}[{old_code}] {'/'.join(old_path_block) or '<root>'}  -->  "
                f"stage{self.n_stage}[{new_code}] {'/'.join(new_path_block) or '<root>'}: "
                f"{len(trasplant)}/{n_new} target arrays filled "
                f"({len(trasplant)}/{n_old} source arrays used, {moved} values)"
            )
            if len(trasplant) < n_new:
                copied = {new_rel for _, new_rel in trasplant}
                for path, leaf in flatten_with_paths(new_params_block):
                    if path not in copied:
                        print(
                            f"    random init: {'/'.join(map(str, new_path_block + path))} "
                            f"{tuple(leaf.shape)}"
                        )

        if self.load:
            print("**************************************************************\n")

        # Previous-stage params are only needed once
        self.prev_params = None

        return new_params

    def load_vstate(self, vstate):

        paths = [resolve_load_path(p) for p in self.load_model["load_from"]]
        checkpoint = self.load_model["checkpoint"]

        assert all(
            [
                any(
                    [
                        tag in path
                        for tag in ["checkpoint", ".json", "parameters", "vstate", "_best.orbax"]
                    ]
                )
                for path in paths
            ]
        )

        if self.load_model["parameters"]:
            for i, path in enumerate(paths):
                self.params_history[f"stageX{i}"] = upgrade_legacy_params(
                    load_from_file(path, checkpoint=checkpoint, tag="parameters")
                )

            vstate.parameters = self.weight_transplantation(vstate.parameters)
            self.params_history = {}

        if self.load_model["sampler"]:  # vstate.sampler_state
            sampler_data = load_from_file(
                paths[0], checkpoint=checkpoint, tag="sampler_state"
            )

            sampler_state = MetropolisSamplerState(
                σ=sampler_data["σ"],
                rng=sampler_data["rng"],
                rule_state=sampler_data["rule_state"],
                log_prob=sampler_data["log_prob"],
            )
            # print(sampler_data, '\n')
            # print(sampler_state, '\n')
            vstate.sampler_state = sampler_state
            # print(vstate.sampler_state)

        return vstate
