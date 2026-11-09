"""
EnergyPlotter: live view of the training (energy, V-score, error).

Panels (global step on the x axis, stage changes as vertical lines):
    1. energy of every step (light) + running mean over `window` steps,
       best running mean so far, ED energies of the sector and global
    2. V-score (log)
    3. relative error of the running mean w.r.t. the ED energy of the
       sector (log); without ED, the error of the mean of every step

It refreshes the window every `refresh_each` steps when matplotlib has an
interactive backend (QtAgg with a display). Otherwise, or in addition, it
overwrites `<sim>_live_energy.png` every `savefig_each` steps and at the end.
With neither a display nor saving it switches itself off (no cost).

config0 "energy_plot_setup":
    refresh_each   steps between redraws of the window (default 10)
    savefig_each   steps between PNG writes (default 0 = only at the end)
    savefig        old flag: true = savefig_each 100
    window         running-mean window (default 50)
    max_points     points drawn per curve, older ones are thinned (default 4000)
    interactive    force (true/false) the live window; default: auto (GUI backend)
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np


def _interactive():
    return matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo", "template")


class EnergyPlotter:
    def __init__(self, N, setup=None, E_ref=None, E_global=None, out_path=None, title=""):
        setup = setup or {}
        self.N = N
        self.E_ref, self.E_global = E_ref, E_global
        self.refresh_each = int(setup.get("refresh_each", 10))
        se = setup.get("savefig_each")
        if se is None:
            se = 100 if setup.get("savefig", False) else 0
        self.savefig_each = int(se)
        self.window = int(setup.get("window", 50))
        self.max_points = int(setup.get("max_points", 4000))
        self.out_path = out_path
        self.interactive = bool(setup.get("interactive", _interactive()))
        self.enabled = self.interactive or (out_path is not None)
        if not self.enabled:
            print("[EnergyPlotter] no display and no output path: disabled")
            return
        self.title = title
        self.step = -1
        self.E, self.err, self.V = [], [], []
        self.stage_lines = []
        self.stage_starts = [0]  # the running mean restarts at every stage
        self._build()

    # ------------------------------------------------------------------

    def _build(self):
        import matplotlib.pyplot as plt

        if self.interactive:
            plt.ion()
        self.fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        self.ax_e, self.ax_v, self.ax_r = axs
        (self.l_e,) = self.ax_e.plot([], [], ".", ms=2, color="tab:blue", alpha=0.35, label="E (paso)")
        (self.l_em,) = self.ax_e.plot([], [], "-", lw=1.6, color="tab:blue", label=f"media {self.window}")
        (self.l_best,) = self.ax_e.plot([], [], "*", ms=10, color="gold", mec="k", label="mejor media")
        if self.E_ref is not None:
            self.ax_e.axhline(self.E_ref, color="tab:green", lw=1.2, label="ED (sector)")
        if self.E_global is not None and self.E_global != self.E_ref:
            self.ax_e.axhline(self.E_global, color="tab:green", ls="--", lw=1, label="ED (global)")
        (self.l_v,) = self.ax_v.plot([], [], "-", lw=0.8, color="tab:purple")
        (self.l_r,) = self.ax_r.plot([], [], "-", lw=1.0, color="tab:red")
        self.ax_e.set_ylabel("Energía")
        self.ax_v.set_ylabel("V-score")
        self.ax_v.set_yscale("log")
        self.ax_r.set_ylabel("Error relativo" if self.E_ref is not None else "σ(E) paso")
        self.ax_r.set_yscale("log")
        self.ax_r.set_xlabel("paso")
        for ax in (self.ax_e, self.ax_v, self.ax_r):
            ax.grid(alpha=0.3)
        self.ax_e.legend(fontsize=7, loc="upper right")
        self.fig.suptitle(self.title, fontsize=9)
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _running_mean(self, x):
        """Running mean over `window` steps, restarted at every stage."""
        x = np.asarray(x, float)
        out = np.empty_like(x)
        bounds = self.stage_starts + [len(x)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = x[a:b]
            if seg.size == 0:
                continue
            w = min(self.window, seg.size)
            c = np.cumsum(np.insert(seg, 0, 0.0))
            o = np.empty_like(seg)
            o[w - 1:] = (c[w:] - c[:-w]) / w
            o[: w - 1] = c[1:w] / np.arange(1, w)
            out[a:b] = o
        return out

    def _thin(self, x, y):
        n = len(x)
        if n <= self.max_points:
            return x, y
        k = int(math.ceil(n / self.max_points))
        return x[::k], y[::k]

    # hooks -------------------------------------------------------------

    def new_stage(self, stage=None):
        if self.enabled:
            self.stage_starts.append(self.step + 1)
            ln = self.ax_e.axvline(self.step + 0.5, color="gray", ls=":", lw=1)
            for ax in (self.ax_v, self.ax_r):
                ax.axvline(self.step + 0.5, color="gray", ls=":", lw=1)
            self.stage_lines.append(ln)

    def finalize(self):
        if self.enabled:
            self.redraw(save=True)

    # ------------------------------------------------------------------

    def __call__(self, step, log_data, driver):
        if not self.enabled:
            return True
        self.step += 1
        st = log_data[driver._loss_name]
        e = float(np.real(st.mean))
        var = float(np.real(st.variance))
        self.E.append(e)
        self.V.append(self.N * var / e**2 if e else np.nan)
        self.err.append(float(np.real(st.error_of_mean)))
        save = self.savefig_each > 0 and self.step % self.savefig_each == 0
        if (self.interactive and self.step % self.refresh_each == 0) or save:
            self.redraw(save=save)
        return True

    def redraw(self, save=False):
        n = len(self.E)
        if n == 0:
            return
        x = np.arange(n)
        E = np.asarray(self.E)
        Em = self._running_mean(E)
        self.l_e.set_data(*self._thin(x, E))
        self.l_em.set_data(*self._thin(x, Em))
        # best running mean of the current stage (the one the keeper restores)
        a = self.stage_starts[-1] if self.stage_starts[-1] < n else 0
        w0 = min(a + self.window - 1, n - 1)
        ib = int(np.nanargmin(Em[w0:]) + w0)
        self.l_best.set_data([ib], [Em[ib]])
        self.l_v.set_data(*self._thin(x, np.asarray(self.V)))
        if self.E_ref is not None:
            r = np.abs(Em - self.E_ref) / abs(self.E_ref)
        else:
            r = np.asarray(self.err)
        self.l_r.set_data(*self._thin(x, np.where(r > 0, r, np.nan)))
        # energy axis: ignore the first transient (outside the running mean window)
        lo = max(0, n // 10)
        refs = [v for v in (self.E_ref, self.E_global) if v is not None]
        ys = np.concatenate([E[lo:], refs])
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
        pad = 0.05 * (ymax - ymin + 1e-12)
        self.ax_e.set_ylim(ymin - pad, ymax + pad)
        for ax, y in ((self.ax_v, np.asarray(self.V)), (self.ax_r, r)):
            y = y[np.isfinite(y) & (y > 0)]
            if y.size:  # log axes: limits from the positive data only
                ax.set_ylim(y.min() / 1.5, y.max() * 1.5)
        self.ax_e.set_xlim(0, max(n, 2))
        self.ax_e.set_title(f"paso {n - 1}   E = {Em[-1]:.5f}   mejor media = {Em[ib]:.5f} (paso {ib})",
                            fontsize=9)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        if save and self.out_path:
            self.fig.savefig(self.out_path, dpi=120)
