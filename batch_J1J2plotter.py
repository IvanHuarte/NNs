#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import os

# 1) Fuerza backend no-interactivo ANTES de importar pyplot
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import glob
import json
import re
import shutil
from pathlib import Path
import multiprocessing as mp
import gc

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------
# Config tipográfica/estética
# ---------------------------
fontsize_txt = 12
fontsize_ticks = 15
fontsize_title = 20
fontsize_labels = 15
fontsize_legend = 13


# ---------------------------
# Utilidades
# ---------------------------
def discover_subdirs(parent_folder: str) -> list[str]:
    """Devuelve subdirectorios de simulaciones (contienen 'UUID_')."""
    all_subdirs = []
    for root, dirs, files in os.walk(parent_folder):
        for d in dirs:
            if not d.endswith(".orbax") and ".orbax" not in root:
                full = os.path.join(root, d)
                if "UUID_" in full:
                    all_subdirs.append(full)

    all_subdirs = sorted(set(all_subdirs))
    # Normalizamos a rutas con '/'
    all_subdirs = [d if d.endswith(os.sep) else d + os.sep for d in all_subdirs]
    # Devolvemos rutas relativas a parent (como usabas antes)
    parent_folder = (
        parent_folder if parent_folder.endswith(os.sep) else parent_folder + os.sep
    )
    parent_folder_subdirs = [d.split(parent_folder)[1] for d in all_subdirs]
    return parent_folder_subdirs


def safe_miny(arr: np.ndarray, floor: float = 1e-6) -> float:
    """Devuelve un mínimo positivo para ejes log (evita 0 o negativos)."""
    arr = np.asarray(arr)
    arrp = arr[np.isfinite(arr) & (arr > 0)]
    return float(max(arrp.min() / 2.0, floor)) if arrp.size else floor


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------
# Worker: procesa UNA simulación
# ---------------------------
def process_simulation(args_tuple):
    """
    Renderiza las 3 figuras para una simulación (subdir).
    Esto se ejecuta en un subproceso -> memoria se libera al terminar el worker.
    """
    (parent_folder, write_root, subdir, dpi, figsize_w, figsize_h) = args_tuple

    # Reconstituimos rutas
    write_folder = os.path.join(write_root, subdir)
    ensure_dir(write_folder)

    # Recogemos artifacts JSON de la simulación
    artifact_paths = glob.glob(os.path.join(parent_folder, subdir, "*_results_*.json"))
    if not artifact_paths:
        # Nada que hacer para este subdir
        return subdir

    # Extraemos (J1, J2, path) y ordenamos por J2/J1
    J12_vals_path = []
    for d in artifact_paths:
        # patrón: ... _J1J2_<J1>_<J2>_ ...
        m = re.search(r"_J1J2_([0-9\.]+_[0-9\.]+)_", d)
        if m:
            J1, J2 = m.group(1).split("_")
            try:
                J1f, J2f = float(J1), float(J2)
                if J1f != 0.0:
                    J12_vals_path.append([J1f, J2f, d])
            except ValueError:
                continue

    if not J12_vals_path:
        return subdir

    sorted_J12 = sorted(J12_vals_path, key=lambda x: x[1] / x[0])

    # Inicializamos acumuladores
    results = [[] for _ in range(16)]

    # Recolectamos datos
    last_art = None
    for J1, J2, art_path in sorted_J12:
        with open(art_path, "r") as f:
            art = json.load(f)
        last_art = art  # nos quedamos con el último para metadatos (modelo, tamaño)

        results[0].append(J2 / J1)  # J21
        results[1].append(art["results"]["E_best"])  # E_best
        results[2].append(art["results"]["E_ED"])  # E_ED
        results[3].append(art["results"]["error"])  # error
        results[4].append(art["results"]["vscore"])  # vscore
        results[5].append(art["results"]["S_renyi"])  # S_renyi

        results[6].append(art["results"]["m"])  # m
        results[7].append(art["results"]["ms"])  # ms
        results[8].append(art["results"]["m2"])  # m2
        results[9].append(art["results"]["ms2"])  # ms2

        results[10].append(art["results"]["m_ED"])  # m_ED
        results[11].append(art["results"]["ms_ED"])  # ms_ED
        results[12].append(art["results"]["m2_ED"])  # m2_ED
        results[13].append(art["results"]["ms2_ED"])  # ms2_ED

        results[14].append(art["results"]["fidelity"])  # fidelity
        results[15].append(art["results"]["time_exe"])  # timexe

        # Copia callback de la corrida especial J2/J1 = 0.5 (con tolerancia numérica)
        ratio = J2 / J1
        if np.isclose(ratio, 0.5, atol=1e-9):
            cb = art.get("_artifacts", {}).get("callback", {}).get("plot", None)
            if cb and os.path.exists(cb):
                try:
                    shutil.copy2(cb, write_folder)
                except Exception:
                    pass

    # Convertimos a arrays
    results = [np.asarray(r) for r in results]
    (
        J21,
        E_best,
        E_ED,
        error,
        vscore,
        S_renyi,
        m,
        ms,
        m2,
        ms2,
        m_ED,
        ms_ED,
        m2_ED,
        ms2_ED,
        fidelity,
        timexe,
    ) = results

    # Elegimos fase segun signo de J2/J1
    m_phase = np.where(J21 < 0, m, ms)
    m_phase_ED = np.where(J21 < 0, m_ED, ms_ED)
    m2_phase = np.where(J21 < 0, m2, ms2)
    m2_phase_ED = np.where(J21 < 0, m2_ED, ms2_ED)

    # Metadatos desde el último artifact
    size = last_art["CM"]["size"]
    size_txt = f"{size[0]}x{size[1]}"
    model_name = last_art["NN"]["name"]
    J1_last = sorted_J12[-1][0]

    # ---------------------------
    #   FIGURA 1
    # ---------------------------
    fig1, ax1 = plt.subplots(3, 1, figsize=(figsize_w, figsize_h))
    ax1[0].set_title(
        r"$%s \qquad J_1=%.2f \quad (%s)$" % (model_name, J1_last, size_txt),
        fontsize=fontsize_title,
    )
    ax1[0].set_ylabel(r"$Energy$", fontsize=fontsize_labels)
    ax1[0].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1[0].plot(
        J21,
        E_best,
        color="blue",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=rf"$E\;({size_txt})$",
    )
    ax1[0].plot(
        J21,
        E_ED,
        color="lime",
        ls="--",
        marker="o",
        ms=3,
        lw=1.5,
        alpha=0.8,
        label=rf"$E_{{ED}}\;({size_txt})$",
    )
    ax1[0].grid()
    ax1[0].legend(fontsize=fontsize_legend)

    y_min = safe_miny(error)
    ax1[1].set_ylim(y_min, 1)
    ax1[1].set_ylabel(r"$rel.\;\;Error$", fontsize=fontsize_labels)
    ax1[1].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1[1].plot(
        J21,
        error,
        color="red",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=rf"{size_txt}",
    )
    ax1[1].set_yscale("log")
    ax1[1].grid()
    ax1[1].legend(fontsize=fontsize_legend)

    y_min = safe_miny(vscore)
    ax1[2].set_ylim(y_min, 1)
    ax1[2].set_ylabel(r"$Vscore$", fontsize=fontsize_labels)
    ax1[2].set_xlabel(r"$J_2 / J_1$", fontsize=fontsize_labels)
    ax1[2].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1[2].plot(
        J21,
        vscore,
        color="purple",
        marker="o",
        ms=3,
        lw=1.5,
        alpha=0.8,
        label=rf"{size_txt}",
    )
    ax1[2].set_yscale("log")
    ax1[2].grid()
    ax1[2].legend(fontsize=fontsize_legend)
    fig1.tight_layout()

    # ---------------------------
    #   FIGURA 2
    # ---------------------------
    fig2, ax2 = plt.subplots(3, 1, figsize=(figsize_w, figsize_h))
    ax2[0].set_title(
        r"$%s \qquad J_1=%.2f \quad (%s)$" % (model_name, J1_last, size_txt),
        fontsize=fontsize_title,
    )
    ax2[0].set_ylabel(r"$S_{renyi}$", fontsize=fontsize_labels)
    ax2[0].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax2[0].plot(
        J21,
        S_renyi,
        color="green",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=rf"{size_txt}",
    )
    ax2[0].grid()
    ax2[0].legend(fontsize=fontsize_legend)

    ax2[1].set_ylim(-1, 1)
    ax2[1].set_ylabel(r"$Magnetization\;(M_z)$", fontsize=fontsize_labels)
    ax2[1].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax2[1].plot(
        J21,
        m_phase,
        color="red",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=rf"{size_txt}",
    )
    ax2[1].plot(
        J21,
        m_phase_ED,
        color="lime",
        ls="--",
        alpha=0.8,
        lw=1.5,
        label=rf"{size_txt}\;(ED)",
    )
    ax2[1].grid()
    ax2[1].legend(fontsize=fontsize_legend)

    ax2[2].set_ylim(0, 1)
    ax2[2].set_ylabel(r"$Fluctuations\;(M_s)$", fontsize=fontsize_labels)
    ax2[2].set_xlabel(r"$J_2 / J_1$", fontsize=fontsize_labels)
    ax2[2].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax2[2].plot(
        J21,
        m2_phase,
        color="purple",
        marker="o",
        ms=3,
        lw=1.5,
        alpha=0.8,
        label=rf"{size_txt}",
    )
    ax2[2].plot(
        J21,
        m2_phase_ED,
        color="lime",
        ls="--",
        lw=1.5,
        alpha=0.8,
        label=rf"{size_txt}\;(ED)",
    )
    ax2[2].grid()
    ax2[2].legend(fontsize=fontsize_legend)
    fig2.tight_layout()

    # ---------------------------
    #   FIGURA 3
    # ---------------------------
    fig3, ax3 = plt.subplots(3, 1, figsize=(figsize_w, figsize_h))
    ax3[0].set_ylim(0.5, 1.05)
    ax3[0].set_title(
        r"$%s \qquad J_1=%.2f \quad (%s)$" % (model_name, J1_last, size_txt),
        fontsize=fontsize_title * 2 / 3,
    )
    ax3[0].set_ylabel(r"$Fidelity$", fontsize=fontsize_labels * 2 / 3)
    ax3[0].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax3[0].plot(
        J21,
        fidelity,
        color="darkorange",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=rf"{size_txt}",
    )
    ax3[0].grid()
    ax3[0].legend(fontsize=fontsize_legend * 2 / 3)

    ax3[1].set_ylabel(r"$TimeExe\;('')$", fontsize=fontsize_labels * 2 / 3)
    ax3[1].set_xlabel(r"$J_2 / J_1$", fontsize=fontsize_labels * 2 / 3)
    ax3[1].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax3[1].plot(
        J21,
        timexe,
        color="olivedrab",
        alpha=0.8,
        marker="o",
        ms=3,
        lw=1.5,
        label=rf"{size_txt}",
    )
    ax3[1].grid()
    ax3[1].legend(fontsize=fontsize_legend * 2 / 3)

    # Learning rate schedule (import local aquí para no cargar el módulo si no hace falta)
    from NN_module.schedule.utils import calculate_lr_schedule

    lr, info = calculate_lr_schedule(last_art["SIM"]["schedule"]["learning_rate"])
    lr_concat = np.concatenate(lr, axis=-1)
    total_epochs = last_art["SIM"]["schedule"]["total_epochs"]

    ax3[2].set_xlabel(r"$Epochs$", fontsize=fontsize_labels)
    ax3[2].set_ylabel(r"$Learning\;Rate$", fontsize=fontsize_labels)
    ax3[2].plot(range(total_epochs), lr_concat)
    ax3[2].set_yscale("log")
    ax3[2].grid(which="both", ls="--")

    cum = 0
    for inf in info:
        x_pos = (cum + inf[0]) / 2
        label = f"{inf[2]}\n{inf[1]}"
        ax3[2].axvline(x_pos * 2, color="grey", linestyle="--")
        ax3[2].text(
            x_pos,
            0.95,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            transform=ax3[2].get_xaxis_transform(),
        )
        cum += inf[0]
    fig3.tight_layout()

    # Guardamos
    fig1.savefig(
        os.path.join(write_folder, f"RunInJ_J1_{J1_last}_Energy_Error_Vscore.jpeg"),
        dpi=dpi,
        bbox_inches="tight",
    )
    fig2.savefig(
        os.path.join(write_folder, f"RunInJ_J1_{J1_last}_Renyi_Mz_Ms.jpeg"),
        dpi=dpi,
        bbox_inches="tight",
    )
    fig3.savefig(
        os.path.join(write_folder, f"RunInJ_J1_{J1_last}_Fidelity_Timexe_lr.jpeg"),
        dpi=dpi,
        bbox_inches="tight",
    )

    # Limpiamos agresivamente
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    del fig1, fig2, fig3, ax1, ax2, ax3, results, last_art
    gc.collect()

    return subdir


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Directorio padre con simulaciones",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="/home/ihuarte/Escritorio/Ivan/NNs/Results/",
        help="Directorio raíz de salida",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Número de procesos concurrentes (ajusta según RAM)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI de las figuras (por defecto 300)"
    )
    parser.add_argument(
        "--figw", type=float, default=10.0, help="Anchura figura (inches)"
    )
    parser.add_argument(
        "--figh", type=float, default=16.0, help="Altura figura (inches)"
    )
    args = parser.parse_args()

    parent = args.directory
    parent_folder = parent if parent.endswith(os.sep) else parent + os.sep

    write_root = args.out if args.out.endswith(os.sep) else args.out + os.sep
    target = parent.split(str(Path(parent).parent) + os.sep)[1]
    write_root = os.path.join(write_root, target)
    ensure_dir(write_root)

    # Descubrimos simulaciones
    subdirs = discover_subdirs(parent_folder)
    if not subdirs:
        print("No se encontraron subdirectorios de simulación.")
        return

    # Empaquetamos argumentos por simulación
    job_args = [
        (parent_folder, write_root, sd, args.dpi, args.figw, args.figh)
        for sd in subdirs
    ]

    print(f"Procesando {len(job_args)} simulaciones con {args.jobs} procesos...")

    # IMPORTANTE: usa 'spawn' para aislamiento total de memoria
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Ya estaba establecido: no pasa nada
        pass

    with mp.Pool(processes=args.jobs) as pool:
        for i, done in enumerate(pool.imap_unordered(process_simulation, job_args), 1):
            print(f"[{i}/{len(job_args)}] OK: {done}")

    print("Terminado.")


if __name__ == "__main__":
    main()
