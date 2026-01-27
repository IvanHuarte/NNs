from .BestIterKeeper import BestIterKeeper
from .EnergyPlotter import EnergyPlotter
from .ModPhasePlotter import ModPhasePlotter
from .SanityMonitor import SanityMonitor


def Callback(
    config, sim_config=None, total_epochs=None, H=None, N=None, E_ED=None, x_ED=None
):

    callback_funcs = []
    callback_objects = []

    if config["keeper"]:
        assert all([variable is not None for variable in [total_epochs, H, N]])
        print(f"Adding BestIterKeeper")
        keeper = BestIterKeeper(total_epochs, H, N, baseline=1e-8, mode="always")
        callback_objects.append(keeper)
        callback_funcs.append(keeper.update)

    if config["energy_plot"]:
        assert all([variable is not None for variable in [H, N]])
        print(f"Adding EnergyPlotter")
        energy_plotter = EnergyPlotter(H, N, E_ED=E_ED)
        callback_objects.append(energy_plotter)
        callback_funcs.append(energy_plotter)

    if config["modphase"]:
        assert all([variable is not None for variable in [sim_config, x_ED]])
        print(f"Adding Modphase")
        modphase = ModPhasePlotter(sim_config, x_ED)
        callback_objects.append(modphase)
        callback_funcs.append(modphase)

    if config["sanity"]:
        print(f"Adding SanityMonitor")
        sanity_monitor = SanityMonitor(config["callback"]["sanity_setup"])
        callback_objects.append(sanity_monitor)
        callback_funcs.append(sanity_monitor)

    return callback_objects, callback_funcs
