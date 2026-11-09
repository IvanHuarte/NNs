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
):

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
            baseline=ks["baseline"],
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

    if config["energy_plot"]:
        assert all([variable is not None for variable in [H, N]])
        print("Adding EnergyPlotter")
        energy_plotter = EnergyPlotter(
            H,
            N,
            E_prev=E_prev,
            E_ED=E_ED,
            vs_prev=vs_prev,
            error_prev=error_prev,
            sim_folder=sim_folder,
            savefig=config["energy_plot_setup"]["savefig"],
        )
        callback_objects.append(energy_plotter)
        callback_funcs.append(energy_plotter)

    if config["modphase"]:
        # assert all([variable is not None for variable in [sim_config, x_ED]])
        print("Adding Modphase")
        modphase = ModPhasePlotter(
            sim_config,
            x_ED,
            no_null_mod=config["modphase_setup"]["no_null_mod"],
            plot_each=config["modphase_setup"]["plot_each"],
            savefig=config["modphase_setup"]["savefig"],
            logscale=config["modphase_setup"]["logscale"],
            sim_folder=sim_folder,
        )
        callback_objects.append(modphase)
        callback_funcs.append(modphase)

    if config["sanity"]:
        print("Adding SanityMonitor")
        sanity_monitor = SanityMonitor(config["sanity_setup"])
        callback_objects.append(sanity_monitor)
        callback_funcs.append(sanity_monitor)

    if config["leaves_gradient"]:
        print("Adding LeavesGradient")
        leaves_gradient = LeavesGradient(config["leaves_gradient_setup"])
        callback_objects.append(leaves_gradient)
        callback_funcs.append(leaves_gradient)

    if config["checkpoint"]:
        print("Adding OrbaxCheckpointing")
        checkpointing = Checkpoint(
            H=H, sim_label_folder=sim_label_folder, setup=config["checkpoint_setup"]
        )
        callback_objects.append(checkpointing)
        callback_funcs.append(checkpointing)

    return callback_objects, callback_funcs
