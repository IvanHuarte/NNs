import os
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from NN_module.label_utils import get_filenames_from_settings
from NN_module.observables import modphase


def plot_modphase(artifact, modphase_vs, step=None, modphase_ED=None):
    """

    Returns a plot of module and phase given a variational state

    """

    (mod_vs, phase_vs), stats_vs = modphase_vs
    if modphase_ED is not None:
        (mod_ED, phase_ED), stats_ED = modphase_ED

    if artifact is not None:
        _, _, _, callback = get_filenames_from_settings(
            artifact["CM"], {"name": artifact["NN"]["name"], "setup": {}}
        )
    else:
        callback = "Callback"

    title = callback.replace("Callback", "").lstrip().replace(" ", "\\quad")

    title += f" (step {step})" if step is not None else ""

    # Plot
    # If ED exists
    if modphase_ED is not None:

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


def plot_modphase_from_vstate(vstate, artifact, x_ED=None, step=None):
    """

    Returns a plot of module and phase given a variational state

    """

    modphase_vs = modphase(vstate)
    if x_ED is not None:
        modphase_ED = modphase(x_ED)

    fig, ax = plot_modphase(artifact, modphase_vs, step=step, modphase_ED=modphase_ED)

    return fig, ax


def single_modphase_plot(x, label, write_folder):

    (mod, phase), stats = modphase(x)

    fig, ax = plt.subplots(3, 1, figsize=[20, 10])

    ax[0].set_title(r"$Modulus\;and\;Phase\qquad %s$" % (label), fontsize=15)
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"$Modulus$")
    ax[0].set_ylim(-0.00001, max(max(mod), max(mod)) * 9 / 8)
    ax[0].plot(mod, alpha=0.6, color="r", label=f"vstate")
    ax[0].legend()

    ax[1].set_xticks([])
    ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax[1].set_ylabel(r"$Phase \;vstate$")
    ax[1].set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax[1].plot(phase, alpha=0.15, ls="", marker="o", ms=0.1, color="r", label="vstate")
    ax[1].set_xlabel(r"$C_i$")
    ax[1].legend(loc="upper right")

    ax[2].set_xlabel(r"$Phase\;(radians)$")
    ax[2].set_ylabel(r"$Phase \;histogram$")

    ax[2].hist(
        phase,
        bins=1000,
        range=(-np.pi, np.pi),
        color="r",
        density=True,
        alpha=0.7,
        label=f"vstate ({stats['type']})",
    )
    ax[2].text(
        0.8,
        0.7,
        r"$\varphi=%.2f \pm %.2f$" % (stats["phase"]["mean"], stats["phase"]["std"]),
        transform=ax[2].transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.4),
    )

    transform = mtransforms.blended_transform_factory(ax[2].transData, ax[2].transAxes)
    if stats["peaks"] is not None:
        for peak in stats["peaks"]["values"]:
            ax[2].text(
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
    if write_folder is not None:
        os.makedirs(write_folder, exist_ok=True)
        plt.savefig(write_folder + label + ".jpeg", dpi=600, bbox_inches="tight")

    return fig, ax


def batch_modphase_plotter(
    x_batch, labels=None, write_folder=None, return_figure=False
):

    if labels is None:
        labels = [f"State\;\#{i}" for i in range(len(x_batch))]

    if return_figure:
        for i, x in enumerate(x_batch):
            yield single_modphase_plot(x, labels[i], write_folder=write_folder)
    else:
        for i, x in enumerate(x_batch):
            fig, _ = single_modphase_plot(x, labels[i], write_folder=write_folder)
            fig.show()


def plot_M_sector_hist(x, hilbert):

    N = hilbert.size

    mod = jnp.abs(x)
    configs = hilbert.all_states()

    max_contr_idx = jnp.argsort(mod)[::-1]
    sorted_modulus = mod[max_contr_idx]
    sorted_configs = configs[max_contr_idx, :]

    config_magnetization = jnp.sum(sorted_configs, axis=-1) / 2

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=[20, 10])

    ax[0].set_title("Magnetization histogram (raw histogram)")
    ax[1].set_title("Magnetization histogram (weighted by modulus)")
    ax[0].hist(
        config_magnetization,
        bins=N,
        range=(-N // 2 - 0.5, N // 2 - 0.5),
        density=True,
        alpha=0.7,
        label=f"ED",
        edgecolor="black",
    )
    ax[1].hist(
        config_magnetization,
        bins=N,
        range=(-N // 2 - 0.5, N // 2 - 0.5),
        weights=sorted_modulus,
        density=True,
        alpha=0.7,
        label=f"ED",
        edgecolor="black",
    )

    ax[0].set_xticks(jnp.arange(-N // 2 - 1, N // 2 + 2, 1))
    ax[1].set_xticks(jnp.arange(-N // 2 - 1, N // 2 + 2, 1))
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[0].set_ylim(1e-2, 1e2)
    # ax[1].set_ylim(1e-2, 1e2)

    return fig, ax


def plot_cum_contributors(x):

    mod = jnp.abs(x)
    min_contr = int(jnp.log(jnp.min(mod)))
    max_contr_idx = jnp.argsort(mod)[::-1]
    sorted_modulus = mod[max_contr_idx]

    tramos = jnp.logspace(min_contr, 0, 100)
    perc_under_x = []

    for x in tramos:
        perc_under_x.append((sorted_modulus < x).sum() / sorted_modulus.shape[0] * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tramos, perc_under_x)
    ax.set_title("Percentage of total components under modulus ", fontsize=20)
    ax.set_xlabel("Modulus", fontsize=15)
    ax.set_ylabel("Percentage", fontsize=15)
    ax.set_xscale("log")

    return fig, ax


def plot_weighted_phase_hist(x):

    mod = jnp.abs(x)
    phase = jnp.angle(x)

    no_null_mod_mask = mod > 1e-12
    phase_mask = phase[no_null_mod_mask]

    fig, ax = plt.subplots(3, 1, figsize=[20, 10])
    ax[0].set_title("Phase histogram")
    ax[1].set_title("Phase histogram with no null modulus")
    ax[2].set_title("Phase histogram weighted by modulus")
    ax[0].hist(
        phase,
        bins=1000,
        range=(-np.pi, np.pi),
        density=True,
        alpha=0.7,
        label=f"ED",
    )
    ax[1].hist(
        phase_mask,
        bins=1000,
        range=(-np.pi, np.pi),
        density=True,
        alpha=0.7,
        label=f"ED",
    )
    ax[2].hist(
        phase,
        bins=1000,
        range=(-np.pi, np.pi),
        weights=mod,
        density=True,
        alpha=0.7,
        label=f"ED",
    )

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[0].set_ylim(1e-2, 1e2)
    ax[1].set_ylim(1e-2, 1e2)
    ax[2].set_ylim(1e-2, 1e2)

    return fig, ax
