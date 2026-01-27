from .BestIterKeeper import BestIterKeeper
from .EnergyPlotter import EnergyPlotter
from .ModPhasePlotter import ModPhasePlotter
from .SanityMonitor import SanityMonitor


def Callback(
    config, sim_config=None, total_epochs=None, H=None, N=None, E_ED=None, x_ED=None
):

    callbacks = []

    if config["keeper"]:
        assert all([variable is not None for variable in [total_epochs, H, N]])
        print(f"Adding BestIterKeeper")
        keeper = BestIterKeeper(total_epochs, H, N, baseline=1e-8, mode="always")
        callbacks.append(keeper.update)

    if config["energy_plot"]:
        assert all([variable is not None for variable in [H, N]])
        print(f"Adding EnergyPlotter")
        inline_energy = EnergyPlotter(H, N, E_ED=E_ED)
        callbacks.append(inline_energy)

    if config["modphase"]:
        assert all([variable is not None for variable in [sim_config, x_ED]])
        print(f"Adding Modphase")
        inline_modphase = ModPhasePlotter(sim_config, x_ED)
        callbacks.append(inline_modphase)

    if config["sanity"]:
        print(f"Adding SanityMonitor")
        sanity_monitor = SanityMonitor(config["callback"]["sanity_setup"])
        callbacks.append(sanity_monitor)

    return callbacks
