"""
Console summary of a simulation (rich panels).

Shows what will actually run: the resolved sampler block, the real number of
ranks/samples, one row per schedule period, optimizer/VMC and the coupling
model parameters. The architecture itself is printed by Hydra (tree + diagram).
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from Hydra.label_utils import get_string_from_nnsetup

MAX_STR = 60
SIM_COLOR = "cyan"
CM_COLOR = "red"
NN_COLOR = "bright_yellow"


def _short(v, n=MAX_STR):
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def _kv_table(items, n_cols):
    """key=value grid with n_cols pairs per row."""
    table = Table.grid(padding=(0, 3))
    for _ in range(n_cols):
        table.add_column()
    items = list(items)
    for i in range(0, len(items), n_cols):
        row = [f"[bold]{k}[/]= {_short(v)}" for k, v in items[i : i + n_cols]]
        table.add_row(*row, *[""] * (n_cols - len(row)))
    return table


def _sampler_panel(sampler_setup, n_samples, n_ranks):
    selection = sampler_setup.get("selection", "")
    block = sampler_setup.get(selection) or {}
    rules = block.get("sampler", {})
    items = [("selection", selection)]
    items += [(k, v) for k, v in rules.items()]
    items += [
        ("n_chains_per_rank", sampler_setup.get("n_chains_per_rank")),
        ("n_samples_per_chain", sampler_setup.get("n_samples_per_chain")),
        ("n_ranks", n_ranks),
        ("n_samples", n_samples),
        ("chunk_vstate", sampler_setup.get("chunk_vstate")),
        ("chunk_sampler", sampler_setup.get("chunk_sampler")),
    ]
    return Panel(_kv_table(items, 2), title="[bold cyan]Sampler[/]", border_style=SIM_COLOR)


def _schedule_panel(schedule, ds_info):
    table = Table(box=None, pad_edge=False, header_style="bold")
    for col in ("#", "stage", "epochs", "modes", "lr", "diag_shift"):
        table.add_column(col)

    i = 0
    for eon, (eon_epo, eon_mode, eon_lr) in enumerate(
        zip(schedule.epochs_struct, schedule.modes_struct, schedule.lr_struct)
    ):
        for era_epo, era_mode, era_lr in zip(eon_epo, eon_mode, eon_lr):
            for epo, mode, lr in zip(era_epo, era_mode, era_lr):
                table.add_row(
                    str(i + 1),
                    str(eon),
                    str(epo),
                    " ".join(mode),
                    _short("  ".join(map(str, lr)), 40),
                    ds_info[i] if i < len(ds_info) else "",
                )
                i += 1

    title = f"[bold cyan]Schedule[/] ({schedule.total_epochs} epochs"
    title += f", rescale {schedule.rescale})" if schedule.rescale != 1.0 else ")"
    return Panel(table, title=title, border_style=SIM_COLOR)


def _optimizer_panel(optimizer_setup, vmc_setup):
    items = [("optimizer", optimizer_setup.get("optimizer"))]
    mods = optimizer_setup.get("modifications") or {}
    items += [(k, v) for k, v in mods.items() if v is not None]
    selection = vmc_setup.get("selection")
    items += [("VMC", selection)]
    items += [(k, v) for k, v in (vmc_setup.get(selection) or {}).items()]
    return Panel(
        _kv_table(items, 1), title="[bold cyan]Optimizer / VMC[/]", border_style=SIM_COLOR
    )


def _cm_panel(cm_setup):
    items = list(cm_setup.get("params", {}).items())
    items += list(cm_setup.get("kwargs_lattice", {}).items())
    items += [("S_operators", cm_setup.get("S_operators"))]
    return Panel(
        _kv_table(items, 4),
        title=f"[bold]COUPLING MODEL ({cm_setup['name']})[/]",
        border_style=CM_COLOR,
    )


def _nn_panel(nn_name, nn_evol_setup):
    table = Table(box=None, pad_edge=False, header_style="bold")
    for col in ("stage", "template", "load", "other"):
        table.add_column(col)
    for stage, setup in nn_evol_setup.items():
        load = setup.get("load")
        load = "  ".join("{}:{}→{}".format(*l) for l in load) if load else ""
        other = [k for k in setup if k not in ("template", "load") and setup[k]]
        table.add_row(
            stage,
            _short(get_string_from_nnsetup(setup.get("template", {}))),
            load,
            " ".join(other),
        )
    return Panel(table, title=f"[bold]NEURAL NETWORK ({nn_name})[/]", border_style=NN_COLOR)


def display_simulation_settings(
    *,
    cm_setup,
    nn_name,
    nn_evol_setup,
    sampler_setup,
    n_samples,
    n_ranks,
    schedule,
    ds_info,
    optimizer_setup,
    vmc_setup,
    console=None,
):
    """
    Print the simulation summary.

    sampler_setup: resolved block (resolve_sampler_setup).
    schedule / ds_info: the Schedule object and its diag_shift info strings.
    """
    console = console or Console()
    size = "x".join(map(str, cm_setup["size"]))

    console.rule(
        f"[bold blue]🧲 CM:[/] [{CM_COLOR}]{cm_setup['name']}[/]   "
        f"[bold blue]🧠 NN:[/] [{NN_COLOR}]{nn_name}[/]   "
        f"[bold blue]🧱 Size:[/] [cyan]{size}[/]"
    )

    top = Table.grid(padding=(0, 1))
    top.add_column()
    top.add_column()
    top.add_row(
        _sampler_panel(sampler_setup, n_samples, n_ranks),
        _optimizer_panel(optimizer_setup, vmc_setup),
    )
    console.print(
        Panel(top, title="[bold]SIMULATION[/]", border_style=SIM_COLOR, expand=False)
    )
    console.print(_schedule_panel(schedule, ds_info))
    console.print(_cm_panel(cm_setup))
    console.print(_nn_panel(nn_name, nn_evol_setup))
    console.rule(style="green")
