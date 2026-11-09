"""
ModPhasePlotter: live view of the modulus and phase of the wave function.

Exact mode (Hilbert space with at most `max_states` configurations):
ψ = vstate.to_array() (normalised). With an ED vector of the same sector:
  - ψ is multiplied by exp(-i·arg⟨ψ_ED|ψ⟩), removing the arbitrary global
    phase, so phases can be compared point by point;
  - the fidelity |⟨ψ_ED|ψ⟩| is shown in the title.
Panels: |ψ_i| (and |ψ_ED,i|) per basis configuration, with the sampled
configurations as a rug; phase per configuration (network and ED); phase
histogram weighted by |ψ|² (the physically meaningful distribution) for
the network and ED. `order: "ed"` sorts the configurations by |ψ_ED|.

Sample mode (larger systems): only the current samples are used, which are
distributed as |ψ|²: histograms of log|ψ|² and of the phase over samples.

The figure is built once; every update only replaces the data of the
existing artists (no clearing, one LineCollection for the rug).

config0 "modphase_setup":
    plot_each     steps between updates (default 50)
    savefig_each  steps between PNG writes (default 0 = only at the end);
                  the old flag "savefig": true means every update
    max_states    largest Hilbert space for the exact mode (default 65536)
    order         "basis" (default) or "ed"
    no_null_mod   drop configurations with |ψ_ED| < 1e-12 (default true)
    logscale      logarithmic modulus axis (default false)
    bins          histogram bins (default 200)
    interactive   force (true/false) the live window; default: auto (GUI backend)
"""

from __future__ import annotations

import numpy as np

from ..EnergyPlotter.EnergyPlotter import _interactive


class ModPhasePlotter:
    def __init__(self, setup=None, x_ED=None, out_path=None, title=""):
        setup = setup or {}
        self.plot_each = int(setup.get("plot_each", 50))
        se = setup.get("savefig_each")
        if se is None:
            se = self.plot_each if setup.get("savefig", False) else 0
        self.savefig_each = int(se)
        self.max_states = int(setup.get("max_states", 65536))
        self.order = setup.get("order", "basis")
        self.no_null_mod = bool(setup.get("no_null_mod", True))
        self.logscale = bool(setup.get("logscale", False))
        self.bins = int(setup.get("bins", 200))
        self.out_path = out_path
        self.title = title
        self.interactive = bool(setup.get("interactive", _interactive()))
        self.enabled = self.interactive or out_path is not None
        if not self.enabled:
            print("[ModPhasePlotter] no display and no output path: disabled")
        self.x_ED = None if x_ED is None else np.asarray(x_ED).ravel()
        self.step = -1
        self.mode = None  # "exact" / "samples", decided at the first call
        self.fig = None

    # ------------------------------------------------------------------

    def _build_exact(self, n):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        if self.interactive:
            plt.ion()
        self.fig, (self.ax_m, self.ax_p, self.ax_h) = plt.subplots(3, 1, figsize=(11, 8))
        ed = self.x_ED
        self.perm = np.argsort(-np.abs(ed)) if (ed is not None and self.order == "ed") else np.arange(n)
        self.inv_perm = np.empty_like(self.perm)
        self.inv_perm[self.perm] = np.arange(n)
        self.mask = np.ones(n, bool)
        if ed is not None and self.no_null_mod:
            self.mask = np.abs(ed) > 1e-12
        x = np.arange(n)
        if ed is not None:
            edp = ed[self.perm]
            self.ax_m.plot(x, np.abs(edp), "-", color="tab:green", lw=0.8, alpha=0.8, label="ED")
            m = self.mask[self.perm]
            self.ax_p.plot(x[m], np.angle(edp)[m], ".", ms=1.5, color="tab:green", alpha=0.8, label="ED",
                           zorder=3)
            h, edges = np.histogram(np.angle(ed[self.mask]), bins=self.bins, range=(-np.pi, np.pi),
                                    weights=np.abs(ed[self.mask]) ** 2, density=True)
            self.ax_h.stairs(h, edges, color="tab:green", label="ED (|ψ|²)")
        (self.l_mod,) = self.ax_m.plot([], [], "-", color="tab:red", lw=0.8, alpha=0.8, label="red")
        self.rug = LineCollection([], colors="0.4", linewidths=0.5, alpha=0.3)
        self.ax_m.add_collection(self.rug)
        (self.l_ph,) = self.ax_p.plot([], [], ".", ms=2, color="tab:red", alpha=0.5, label="red")
        self.h_ph = self.ax_h.stairs(np.zeros(self.bins), np.linspace(-np.pi, np.pi, self.bins + 1),
                                     color="tab:red", label="red (|ψ|²)")
        self.ax_m.set_xlim(-0.5, n - 0.5)
        self.ax_p.set_xlim(-0.5, n - 0.5)
        if self.logscale:
            self.ax_m.set_yscale("log")
        self.ax_m.set_ylabel("|ψ|")
        self.ax_p.set_ylabel("fase")
        self.ax_p.set_ylim(-np.pi - 0.1, np.pi + 0.1)
        self.ax_p.set_xlabel("configuración" + (" (ordenadas por |ψ_ED|)" if self.order == "ed" else ""))
        self.ax_h.set_xlabel("fase")
        self.ax_h.set_ylabel("densidad")
        self.ax_h.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax_h.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
        for ax in (self.ax_m, self.ax_p, self.ax_h):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")
        self.fig.suptitle(self.title, fontsize=9)
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _build_samples(self):
        import matplotlib.pyplot as plt

        if self.interactive:
            plt.ion()
        self.fig, (self.ax_a, self.ax_h) = plt.subplots(2, 1, figsize=(11, 6))
        self.h_amp = self.ax_a.stairs(np.zeros(self.bins), np.linspace(0, 1, self.bins + 1), color="tab:red")
        self.h_ph = self.ax_h.stairs(np.zeros(self.bins), np.linspace(-np.pi, np.pi, self.bins + 1),
                                     color="tab:red")
        self.ax_a.set_xlabel("log|ψ|² − max (muestras)")
        self.ax_a.set_ylabel("densidad")
        self.ax_h.set_xlabel("fase (muestras ~ |ψ|²)")
        self.ax_h.set_ylabel("densidad")
        self.ax_h.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax_h.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
        for ax in (self.ax_a, self.ax_h):
            ax.grid(alpha=0.3)
        self.fig.suptitle(self.title, fontsize=9)
        self.fig.tight_layout()

    # ------------------------------------------------------------------

    def finalize(self):
        if self.fig is not None and self.out_path:
            self.fig.savefig(self.out_path, dpi=120)

    def new_stage(self, stage=None):
        pass

    def __call__(self, step, log_data, driver):
        if not self.enabled:
            return True
        self.step += 1
        if self.step % self.plot_each != 0:
            return True
        vstate = driver.state
        hi = vstate.hilbert
        if self.mode is None:
            exact = hi.is_indexable and hi.n_states <= self.max_states
            self.mode = "exact" if exact else "samples"
            if self.x_ED is not None and exact and self.x_ED.size != hi.n_states:
                self.x_ED = None  # vector of another basis: do not compare
            self._build_exact(hi.n_states) if exact else self._build_samples()
        if self.mode == "exact":
            self._update_exact(vstate)
        else:
            self._update_samples(vstate)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        if self.out_path and self.savefig_each and self.step % self.savefig_each == 0:
            self.fig.savefig(self.out_path, dpi=120)
        return True

    def _update_exact(self, vstate):
        psi = np.asarray(vstate.to_array(normalize=True)).ravel()
        fid = None
        if self.x_ED is not None:
            ov = np.vdot(self.x_ED, psi)
            psi = psi * np.exp(-1j * np.angle(ov))
            fid = abs(ov)
        n = psi.size
        x = np.arange(n)
        p = psi[self.perm]
        self.l_mod.set_data(x, np.abs(p))
        m = self.mask[self.perm]
        self.l_ph.set_data(x[m], np.angle(p)[m])
        idx = np.asarray(vstate.hilbert.states_to_numbers(
            np.asarray(vstate.samples).reshape(-1, vstate.hilbert.size)))
        xs = self.inv_perm[np.unique(idx)]
        top = float(np.abs(p).max())
        self.rug.set_segments([[(i, -0.06 * top), (i, 0.0)] for i in xs])
        if not self.logscale:
            self.ax_m.set_ylim(-0.08 * top, 1.1 * top)
        h, _ = np.histogram(np.angle(psi[self.mask]), bins=self.bins, range=(-np.pi, np.pi),
                            weights=np.abs(psi[self.mask]) ** 2, density=True)
        self.h_ph.set_data(h)
        self.ax_h.relim()
        self.ax_h.autoscale_view(scalex=False)
        ttl = f"paso {self.step}   configuraciones muestreadas: {xs.size}"
        if fid is not None:
            ttl += f"   fidelidad |⟨ψ_ED|ψ⟩| = {fid:.5f}"
        self.ax_m.set_title(ttl, fontsize=9)

    def _update_samples(self, vstate):
        s = np.asarray(vstate.samples).reshape(-1, vstate.hilbert.size)
        logpsi = np.asarray(vstate.log_value(s))
        a = 2 * np.real(logpsi)
        a = a - a.max()
        lo = min(a.min(), -1e-3)
        h, edges = np.histogram(a, bins=self.bins, range=(lo, 0), density=True)
        self.h_amp.set_data(h, edges)
        phase = np.angle(np.exp(1j * np.imag(logpsi)))  # wrapped to (-π, π]
        hp, _ = np.histogram(phase, bins=self.bins, range=(-np.pi, np.pi), density=True)
        self.h_ph.set_data(hp)
        for ax in (self.ax_a, self.ax_h):
            ax.relim()
            ax.autoscale_view()
        self.ax_a.set_title(f"paso {self.step}   {s.shape[0]} muestras   "
                            f"σ[log|ψ|²] = {a.std():.2f}", fontsize=9)
