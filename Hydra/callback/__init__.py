import os

import matplotlib

from .BestIterKeeper import BestIterKeeper
from .Checkpoint import Checkpoint
from .EnergyPlotter import EnergyPlotter
from .ModPhasePlotter import ModPhasePlotter
from .SanityMonitor import SanityMonitor
from .LeavesGradient import LeavesGradient

if os.environ.get("DISPLAY"):
    if "localhost" not in os.environ.get("DISPLAY"):
        matplotlib.use("QtAgg")
    else:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")


def Callback(
    config,
    sim_config=None,
    total_epochs=None,
    H=None,
    N=None,
    E_prev=None,
    E_ED=None,
    x_ED=None,
    vs_prev=None,
    error_prev=None,
    sim_label_folder=None,
    sim_folder=None,
    stage_epochs=None,
    best_label_folder=None,
    E_ref=None,
):
    """
    Build the callbacks enabled in config0 "callback". Returns
    (objects, functions). Objects may implement the hooks new_stage(stage),
    flush(wait) and finalize(), called by VMC_simulation.train at a stage
    change, on a crash and at the end of the training.
    """

    callback_funcs = []
    callback_objects = []

    if config["keeper"]:
        assert all([variable is not None for variable in [total_epochs, H, N]])
        print("Adding BestIterKeeper")

        ks = config["keeper_setup"]
        # On-disk bests (orbax) unless keeper_setup "orbax": false
        orbax_path = None
        if ks.get("orbax", True) and best_label_folder:
            orbax_path = best_label_folder + "_best.orbax"
        keeper = BestIterKeeper(
            total_epochs,
            H,
            N,
            baseline=ks.get("baseline", 1e-30),
            mode=ks["mode"],
            orbax_path=orbax_path,
            save_every=ks.get("save_every", 50),
            start_fraction=ks.get("start_fraction", 0.1),
            vscore_max=ks.get("vscore_max", 0.05),
            best_setup=ks.get("best_setup"),
            stage_epochs=stage_epochs,
        )
        callback_objects.append(keeper)
        callback_funcs.append(keeper.update)

    prefix = best_label_folder or sim_label_folder  # <write folder><sim label>
    title = ""
    if sim_config is not None:
        from Hydra.label_utils import make_title

        title = make_title(sim_config["CM"], sim_config["NN"]["name"]).replace("Callback", "").strip()

    if config.get("energy_plot"):
        print("Adding EnergyPlotter")
        energy_plotter = EnergyPlotter(
            N,
            setup=config.get("energy_plot_setup", {}),
            E_ref=E_ref if E_ref is not None else E_ED,
            E_global=E_ED,
            out_path=(prefix + "_live_energy.png") if prefix else None,
            title=title,
        )
        callback_objects.append(energy_plotter)
        callback_funcs.append(energy_plotter)

    if config.get("modphase"):
        print("Adding ModPhasePlotter")
        modphase = ModPhasePlotter(
            setup=config.get("modphase_setup", {}),
            x_ED=x_ED,
            out_path=(prefix + "_live_modphase.png") if prefix else None,
            title=title,
        )
        callback_objects.append(modphase)
        callback_funcs.append(modphase)

    if config.get("sanity"):
        print("Adding SanityMonitor")
        sanity_monitor = SanityMonitor(
            config.get("sanity_setup", {}),
            N=N,
            E_ref=E_ref if E_ref is not None else E_ED,
            log_path=(prefix + "_sanity.jsonl") if prefix else None,
        )
        callback_objects.append(sanity_monitor)
        callback_funcs.append(sanity_monitor)

    if config.get("leaves_gradient"):
        print("Adding LeavesGradient")
        leaves_gradient = LeavesGradient(
            config.get("leaves_gradient_setup", {}), log_prefix=prefix
        )
        callback_objects.append(leaves_gradient)
        callback_funcs.append(leaves_gradient)

    if config.get("checkpoint"):
        print("Adding Checkpoint")
        checkpointing = Checkpoint(
            H=H,
            sim_label_folder=sim_label_folder or prefix,
            setup=config.get("checkpoint_setup", {}),
            stage_epochs=stage_epochs,
        )
        callback_objects.append(checkpointing)
        callback_funcs.append(checkpointing)

    return callback_objects, callback_funcs
