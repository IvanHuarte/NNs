import json
import os
from datetime import date
from pathlib import Path
from platform import architecture, python_version

import flax
import jax
import netket as nk
import numpy as np
import orbax.checkpoint as ocp
import scipy as sp
from netket.sampler.metropolis import MetropolisSamplerState
from VA_project.initialize_model import ModelFactory

from .NN.NN import NeuralNetwork
from .sampler.sampler import SamplerFactory


def save_pytree(path, pytree):
    """
    Save PyTree with structure via Orbax

    :param path: Absolute path to the target
    :param pytree_dict: Pytree
    """
    cp = ocp.PyTreeCheckpointer()
    jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), pytree)
    cp.save(path, pytree)


def load_pytree(path):
    cp = ocp.PyTreeCheckpointer()
    return cp.restore(path)


def save_results(
    vstate,
    setup,
    x_ED_global=None,
    x_ED_irrep=None,
    modphase=None,
    modphase_ED_global=None,
    modphase_ED_irrep=None,
    write_folder="./",
    sim_label="",
    ED_label="",
    json_label="",
    sim_uuid=None,
):
    """
    Save the results of the simulation.
    Args:
        vstate: The variational state.
        setup: The setup dictionary containing the simulation parameters.
        x_ED: The exact diagonalization results (optional).
        write_folder: The folder to save the results.
        sim_label: A label for the simulation.
    """

    os.makedirs(write_folder, exist_ok=True)

    # Save LEGEND .json in previous dir
    legend = {"NN": setup["NN"], "SIM": setup["SIM"], "CM": setup["CM"]}
    parent = str(Path(write_folder).parent) + "/"

    with open(parent + f"UUID_{sim_uuid}.json", "w") as legendfile:
        json.dump(legend, legendfile, separators=(",", ":"), sort_keys=True, indent=4)

    # Save the variational state parameters
    path_vstate = write_folder + sim_label + "_vstate_parameters.orbax"
    save_pytree(path_vstate, vstate.parameters)
    setup["_artifacts"]["parameters"] = path_vstate

    # Save the vstate sampler state
    path_sampler = write_folder + sim_label + "_vstate_sampler_state.orbax"
    save_pytree(path_sampler, vstate.sampler_state)
    setup["_artifacts"]["sampler_state"] = path_sampler

    # Save exact diagonalization eigenstate
    if x_ED_global is not None:
        path_ED_global = write_folder + ED_label + ".txt"
        if not os.path.isfile(write_folder + ED_label):
            np.savetxt(path_ED_global, x_ED_global)
        setup["_artifacts"]["x_ED_global"] = path_ED_global
        if x_ED_irrep is not None:
            irrep = setup["results"]["irrep_info"]["irrep"]
            path_ED_irrep = write_folder + ED_label + f"irrep_{irrep}.txt"
            if not os.path.isfile(write_folder + ED_label):
                np.savetxt(path_ED_irrep, x_ED_irrep)
            setup["_artifacts"]["x_ED_irrep"] = path_ED_irrep

    # Save modulus and phase from vstate and/or xED
    if not "modphase" in setup["_artifacts"]:
        setup["_artifacts"]["modphase"] = {}
    if modphase is not None:
        modphase_path = write_folder + sim_label + "_modphase_vstate.txt"
        np.savetxt(modphase_path, modphase)
        setup["_artifacts"]["modphase"]["vstate"] = modphase_path

    if modphase_ED_global is not None:
        modphase_ED_path_global = write_folder + ED_label + "_modphase_xED_global.txt"
        np.savetxt(modphase_ED_path_global, modphase_ED_global)
        setup["_artifacts"]["modphase"]["xED_global"] = modphase_ED_path_global
        if modphase_ED_irrep is not None:
            modphase_ED_path_irrep = (
                write_folder + ED_label + f"_modphase_xED_irrep_{irrep}.txt"
            )
            np.savetxt(modphase_ED_path_irrep, modphase_ED_irrep)
            setup["_artifacts"]["modphase"]["xED_irrep"] = modphase_ED_path_irrep

    # Write metadata in main setup artifact
    metadata = {
        "uuid": sim_uuid,
        "date": date.today().strftime("%x"),
        "architecture": architecture(),
        "versions": {
            "python": python_version(),
            "numpy": np.__version__,
            "scipy": sp.__version__,
            "netket": nk.__version__,
            "jax": jax.__version__,
            "flax": flax.__version__,
        },
        "device": str(jax.devices()[0]),
    }

    setup["metadata"] = metadata
    setup["results"]["vstate"] = path_vstate

    # Save the main artifact
    setup_path = write_folder + json_label + ".json"
    with open(setup_path, "w") as outfile:
        json.dump(setup, outfile, separators=(",", ":"), sort_keys=True, indent=4)


def print_tree_keys(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{prefix}{key}")
            print_tree_keys(value, indent + 1)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            print(f"{prefix}- [{i}]")
            print_tree_keys(item, indent + 1)


def load_vstate(artifact, params_path=None, sampler_path=None, tree_data=False):

    # Initialize model
    size = artifact["CM"]["size"]
    N = int(np.prod(size))
    cm_model = ModelFactory.init(artifact["CM"]).get_model()

    model = NeuralNetwork(
        artifact["NN"]["setup"], lattice_size=artifact["CM"]["size"]
    ).get_model()

    # Initialize hilbert space
    hi = nk.hilbert.Spin(s=0.5, N=N)

    # # Dummy init to have the vs_params structure
    # rng = jax.random.PRNGKey(0)
    # dummy_params = model.init(rng, jnp.ones((1, N)))

    # Initialize sampler
    sampler = SamplerFactory(artifact["SIM"]["sampler"], cm_model=cm_model).get_sampler(
        hi
    )

    n_samples = artifact["SIM"]["sampler"]["nsamples"]

    # Variational State
    vs = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=0)

    params_path = (
        artifact["_artifacts"]["parameters"] if params_path is None else params_path
    )
    sampler_path = (
        artifact["_artifacts"]["sampler_state"]
        if sampler_path is None
        else sampler_path
    )

    parameters = load_pytree(params_path)
    sampler_data = load_pytree(sampler_path)

    sampler_state = MetropolisSamplerState(
        σ=sampler_data["σ"],
        rng=sampler_data["rng"],
        rule_state=sampler_data["rule_state"],
        log_prob=sampler_data["log_prob"],
    )

    if tree_data:  # For debugging
        params_ok = jax.tree_util.tree_structure(
            parameters
        ) == jax.tree_util.tree_structure(vs.parameters)
        sampler_ok = jax.tree_util.tree_structure(
            sampler_state
        ) == jax.tree_util.tree_structure(vs.sampler_state)
        print(f"Parameter structure{" " if params_ok else " DOESN'T "}fit")
        print(f"Sampler State structure{" " if sampler_ok else " DOESN'T "}fit")

    vs.parameters = parameters
    vs.sampler_state = sampler_state

    return vs
