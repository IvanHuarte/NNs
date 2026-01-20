import os
import json
import numpy as np
import scipy as sp
import flax
import jax
import jax.numpy as jnp
import netket as nk
from flax.serialization import to_bytes, from_bytes
from datetime import date
from platform import architecture, python_version
from pathlib import Path

from VA_project.initialize_model import ModelFactory
from .initialize_NN import FactoryBuilder
from .initialize_sampler import SamplerFactory


def save_results(
    vstate,
    setup,
    x_ED=None,
    modphase=None,
    modphase_ED=None,
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

    path_vstate = write_folder + sim_label + "_vstate_params.msgpack"

    with open(path_vstate, "wb") as f:
        f.write(to_bytes(vstate.parameters))

    setup["_artifacts"]["vstate"] = path_vstate

    # Save the vstate sampler state
    path_sampler = write_folder + sim_label + "_vstate_sampler_state.msgpack"
    with open(path_sampler, "wb") as f:
        f.write(to_bytes(vstate.sampler_state))
    setup["_artifacts"]["sampler_state"] = path_sampler

    # Save exact diagonalization eigenstate
    if x_ED is not None:
        path_ED = write_folder + ED_label + ".txt"
        if not os.path.isfile(write_folder + ED_label):
            np.savetxt(path_ED, x_ED)
        setup["_artifacts"]["x_ED"] = path_ED

    # Save modulus and phase from vstate and/or xED
    if not "modphase" in setup["_artifacts"]:
        setup["_artifacts"]["modphase"] = {}
    if modphase is not None:
        modphase_path = write_folder + sim_label + "_modphase_vstate.txt"
        np.savetxt(modphase_path, modphase)
        setup["_artifacts"]["modphase"]["vstate"] = modphase_path

    if modphase_ED is not None:
        modphase_ED_path = write_folder + ED_label + "_modphase_xED.txt"
        np.savetxt(modphase_ED_path, modphase_ED)
        setup["_artifacts"]["modphase"]["xED"] = modphase_ED_path

    # Write metadata in main setup artifact
    metadata = {
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


def load_vstate(setup, only_parameters, tree_data=False):

    # Initialize model
    size = setup["CM"]["size"]
    N = int(np.prod(size))
    # print(setup["NN"])
    cm_model = ModelFactory.init(setup["CM"]).get_model()

    model = FactoryBuilder(
        setup["NN"]["setup"], **{"lattice_size": setup["CM"]["size"]}
    ).get_model()

    # Initialize hilbert space
    hi = nk.hilbert.Spin(s=0.5, N=N)

    # Dummy init to have the vs_params structure
    rng = jax.random.PRNGKey(0)
    dummy_params = model.init(rng, jnp.ones((1, N)))

    # Initialize sampler
    sampler = SamplerFactory(setup["SIM"]["sampler"], cm_model=cm_model).get_sampler(hi)

    # Load parameters and initialize vstate
    n_samples = (
        setup["SIM"]["sampler"]["n_samples_per_chain"]
        * setup["SIM"]["sampler"]["n_chains_per_rank"]
        * setup["SIM"]["sampler"]["n_ranks"]
    )
    n_samples = n_samples  # setup["SIM"]["sampler"]["n_samples"]

    vs = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=0)
    dummy_params = dummy_params["params"]

    if tree_data:  # For debugging
        import msgpack

        print(f"Parameters structure")
        with open(setup["_artifacts"]["vstate"], "rb") as f:
            data = msgpack.unpack(f, raw=False)
        print_tree_keys(data)

        print(f"Sampler state structure")
        with open(setup["_artifacts"]["sampler_state"], "rb") as f:
            data = msgpack.unpack(f, raw=False)
        print_tree_keys(data)

    # Load params
    vs_path = setup["results"]["vstate"]
    with open(vs_path, "rb") as f:
        loaded_params = from_bytes(dummy_params, f.read())
        vs.parameters = loaded_params

    sampler_path = setup["_artifacts"]["sampler_state"]
    # Load sample_state
    with open(sampler_path, "rb") as f:
        vs.sampler_state = from_bytes(vs.sampler_state, f.read())

    return vs
