from .BestIterKeeper import BestIterKeeper
from .EnergyPlotter import EnergyPlotter
from .ModPhasePlotter import ModPhasePlotter
from .SanityMonitor import SanityMonitor
from .Checkpoint import Checkpoint
import os
import matplotlib

if os.environ.get("DISPLAY"):
    if not "localhost" in os.environ.get("DISPLAY"):
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
):

    callback_funcs = []
    callback_objects = []

    if config["keeper"]:
        assert all([variable is not None for variable in [total_epochs, H, N]])
        print(f"Adding BestIterKeeper")

        keeper = BestIterKeeper(
            total_epochs,
            H,
            N,
            baseline=config["keeper_setup"]["baseline"],
            mode=config["keeper_setup"]["mode"],
        )
        callback_objects.append(keeper)
        callback_funcs.append(keeper.update)

    if config["energy_plot"]:
        assert all([variable is not None for variable in [H, N]])
        print(f"Adding EnergyPlotter")
        energy_plotter = EnergyPlotter(
            H, N, E_prev=E_prev, E_ED=E_ED, vs_prev=vs_prev, error_prev=error_prev
        )
        callback_objects.append(energy_plotter)
        callback_funcs.append(energy_plotter)

    if config["modphase"]:
        # assert all([variable is not None for variable in [sim_config, x_ED]])
        print(f"Adding Modphase")
        modphase = ModPhasePlotter(
            sim_config,
            x_ED,
            no_null_mod=config["modphase_setup"]["no_null_mod"],
            plot_each=config["modphase_setup"]["plot_each"],
            logscale=config["modphase_setup"]["logscale"],
        )
        callback_objects.append(modphase)
        callback_funcs.append(modphase)

    if config["sanity"]:
        print(f"Adding SanityMonitor")
        sanity_monitor = SanityMonitor(config["callback"]["sanity_setup"])
        callback_objects.append(sanity_monitor)
        callback_funcs.append(sanity_monitor)

    if config["checkpoint"]:
        print(f"Adding OrbaxCheckpointing")
        checkpointing = Checkpoint(
            H=H, sim_label_folder=sim_label_folder, setup=config["checkpoint_setup"]
        )
        callback_objects.append(checkpointing)
        callback_funcs.append(checkpointing)

    return callback_objects, callback_funcs
