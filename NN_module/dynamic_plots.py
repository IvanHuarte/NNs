import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from NN_module.label_utils import get_filenames_from_settings
from NN_module.NN_utils import modphase


def plot_modphase_from_vstate(vstate, sim_config, x_ED=None):
    """

    Returns a plot of module and phase given a variational state

    """

    (mod_ED, phase_ED), stats_ED = modphase(x_ED)
    if x_ED is not None:
        (mod_vs, phase_vs), stats_vs = modphase(vstate)

    kwargs = {**sim_config["CM"], **sim_config["NN"]["setup"]}
    _, _, _, callback = get_filenames_from_settings(
        sim_config["CM"]["name"], sim_config["NN"]["name"], **kwargs
    )
    title = callback.replace("Callback", "").lstrip().replace(" ", "\\quad")

    # Plot
    # If ED exists
    if x_ED is not None:

        fig, ax = plt.subplots(4, 1, figsize=[15, 10])

        ax[0].set_title(r"$Modulus\;and\;Phase\qquad %s$" % (title), fontsize=10)
        ax[0].set_xticks([])
        ax[0].set_ylabel(r"$Modulus$")
        ax[0].set_ylim(-0.00001, max(max(mod_ED), max(mod_vs)) * 9 / 8)
        ax[0].plot(mod_vs, alpha=0.6, color="r", label=f"vstate")
        ax[0].plot(mod_ED, alpha=0.6, label=f"ED")
        ax[0].legend()

        ax[1].set_xticks([])
        ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax[1].set_ylabel(r"$Phase \;vstate$")
        ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
        ax[1].plot(
            phase_vs, alpha=0.15, ls="", marker="o", ms=0.1, color="r", label="vstate"
        )
        ax[1].legend(loc="upper right")

        ax[2].set_xlabel(r"$C_i$")
        ax[2].set_ylabel(r"$Phase \;ED$")
        ax[2].set_ylim(-np.pi - 0.1, np.pi + 0.1)
        ax[2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[2].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax[2].plot(phase_ED, alpha=0.15, ls="", marker="o", ms=0.1, label="ED")
        ax[2].legend(loc="upper right")

        ax[3].set_xlabel(r"$Phase\;(radians)$")
        ax[3].set_ylabel(r"$Phase \;histogram$")
        ax[3].hist(
            phase_ED,
            bins=1000,
            range=(-np.pi, np.pi),
            density=True,
            alpha=0.7,
            label=f"ED  ({stats_ED['type']})",
        )
        ax[3].hist(
            phase_vs,
            bins=1000,
            range=(-np.pi, np.pi),
            color="r",
            density=True,
            alpha=0.7,
            label=f"vstate ({stats_vs['type']})",
        )
        ax[3].text(
            0.8,
            0.7,
            r"$\varphi_{ED}=%.2f \pm %.2f$"
            % (stats_ED["phase"]["mean"], stats_ED["phase"]["std"])
            + "\n"
            + r"$\varphi_{vs}=%.2f \pm %.2f$"
            % (stats_vs["phase"]["mean"], stats_vs["phase"]["std"]),
            transform=ax[3].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.4),
        )

        transform = mtransforms.blended_transform_factory(
            ax[3].transData, ax[3].transAxes
        )
        if stats_ED["peaks"] is not None:
            for peak in stats_ED["peaks"]["values"]:
                ax[3].text(
                    peak - 0.1,
                    0.9,
                    r"%.2f" % peak,
                    color="b",
                    transform=transform,
                    fontsize=8,
                    alpha=0.7,
                )

        if stats_vs["peaks"] is not None:
            for peak in stats_vs["peaks"]["values"]:
                ax[3].text(
                    peak - 0.1,
                    0.9,
                    r"%.2f" % peak,
                    color="b",
                    transform=transform,
                    fontsize=8,
                    alpha=0.7,
                )
        ax[3].legend()
        plt.tight_layout()

    else:

        fig, ax = plt.subplots(3, 1, figsize=[13, 9])

        ax[0].set_title(r"$Modulus\;and\;Phase\qquad %s$" % (title), fontsize=10)
        ax[0].set_xticks([])
        ax[0].set_ylabel(r"$Modulus$")
        ax[0].set_ylim(-0.01, max(max(mod_ED), max(mod_vs)) * 9 / 8)
        ax[0].plot(mod_vs, alpha=0.6, color="r", label="vstate")
        ax[0].plot(mod_ED, alpha=0.6, label="ED")
        ax[0].legend()

        ax[1].set_xlabel(r"$C_i$")
        ax[1].set_xticks([])
        ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax[1].set_ylabel(r"$Phase \;vstate$")
        ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
        ax[1].plot(
            phase_vs, alpha=0.25, ls="", marker="o", ms=0.9, color="r", label="vstate"
        )
        ax[1].legend(loc="upper right")

        ax[2].set_xlabel(r"$Phase\;(radians)$")
        ax[2].set_ylabel(r"$Phase \;histogram$")
        ax[2].hist(
            phase_vs,
            bins=1000,
            range=(-np.pi, np.pi),
            color="r",
            density=True,
            alpha=0.7,
            label="vstate",
        )
        ax[3].text(
            0.8,
            0.7,
            r"$\varphi_{vs}=%.2f \pm %.2f$"
            % (stats_ED["mean"], stats_ED["std"], stats_vs["mean"], stats_vs["std"]),
            transform=ax[3].transAxes,
            bbox=dict(facecolor="white", alpha=0.4, fontsize=10),
        )
        transform = mtransforms.blended_transform_factory(
            ax[3].transData, ax[3].transAxes
        )

        if stats_vs["peaks"] is not None:
            for peak in stats_vs["peaks"]["values"]:
                ax[3].text(
                    peak - 0.1,
                    0.9,
                    r"%.2f" % peak,
                    color="b",
                    transform=transform,
                    fontsize=8,
                    alpha=0.7,
                )
        ax[2].legend()
        plt.tight_layout()

    return fig, ax
