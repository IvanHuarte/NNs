import uuid
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

import NN_module.NN
from NN_module.NN import __all_single__, __all_factories__


def get_sim_config(configurations, **kwargs):

    sim = configurations["SIM"]
    cm = configurations["CM"]
    nn = configurations["NN"]
    cleaned = {**kwargs}

    # Simulation
    cleaned["SIM"] = {}

    schedule_setup = sim["schedule"]["learning_rate"]

    cleaned["SIM"]["sampler"] = sim["sampler"]
    cleaned["SIM"]["schedule"] = {**schedule_setup}

    # Coupling model
    cleaned["CM"] = cm

    # NN architecture

    cleaned["NN"] = {}
    cleaned["NN"]["name"] = nn["name"]
    cleaned["NN"]["setup"] = nn["setup"]

    return cleaned


def get_string_from_nnsetup(tree, level=0):
    if not isinstance(tree, dict) or not tree:
        return ""

    names = []

    for module, submodules in tree.items():
        children = get_string_from_nnsetup(submodules, level + 1)
        separator = "_" * (level + 1)

        if children:
            names.append(f"{module}{separator}{children}")
        else:
            names.append(module)

    return "_".join(names)


def get_write_folder_from_model(config):

    name = config["selection"]
    nn_setup = config[name]
    last_stage = next(reversed(nn_setup))
    model_label = config["CM"]["selection"]

    nn_label = get_string_from_nnsetup(nn_setup[last_stage]["template"])
    nn_label = name + "_" + nn_label

    return config["write_folder_sim"] + model_label + "/" + nn_label + "/"


from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def display_simulation_settings(settings, n_cols=5):

    max_str_lenght = 50
    console = Console()

    # Colores para las secciones
    cm_color = "red"
    nn_color = "bright_yellow"
    sim_color = "cyan"

    # Obtener nombres
    cm_name = settings["CM"].get("name", "Unknown")
    nn_name = settings["NN"].get("name", "Unknown")

    # Título principal
    title_text = f"[bold blue]🧲 CM:[/] [{cm_color}]{cm_name}[/]   [bold blue]🧠 NN:[/] [{nn_color}]{nn_name}[/]"
    console.rule(title_text)

    def _group_params(
        params_dict, n_cols=5, exclude_keys=None, max_str_length=max_str_lenght
    ):
        """Agrupa parámetros en filas de n_cols columnas"""
        if exclude_keys is None:
            exclude_keys = set()

        # Filtra claves no relevantes
        items = [
            (k, v)
            for k, v in params_dict.items()
            if not k.endswith("_list") and k not in exclude_keys
        ]

        # Limita la longitud de strings largos
        items = [
            (k, f"{str(v)[:max_str_length]}..." if len(str(v)) > max_str_length else v)
            for k, v in items
        ]

        grouped = [items[i : i + n_cols] for i in range(0, len(items), n_cols)]

        table = Table(show_header=False, box=None, pad_edge=False)
        for i in range(n_cols):
            table.add_column(justify="left")

        for group in grouped:
            row = [f"[bold]{k}[/]= {v}" for k, v in group]
            while len(row) < n_cols:
                row.append("")
            table.add_row(*row)

        return table

    # ============================================================
    # SIZE (encima del panel de SIMULATION)
    # ============================================================
    size = settings.get("size", None) or settings["CM"].get("size", None)
    if isinstance(size, (list, tuple)) and all(isinstance(x, int) for x in size):
        size_str = "x".join(map(str, size))
        console.print(f"[bold yellow]🧱 Size:[/] [cyan]{size_str}[/]\n")

    # ============================================================
    # SIMULATION (con sampler a la izquierda y schedule a la derecha)
    # ============================================================
    sim_dict = settings.get("SIM", {})
    if sim_dict:
        # Información del sampler
        sampler_dict = sim_dict.get("sampler", {})
        sampler_config = sampler_dict.get("sampler", {})
        sampler_info = {
            "rules": sampler_config.get("rules", []),
            "prules": sampler_config.get("prules", []),
            "n_ranks": sampler_dict.get("n_ranks", ""),
            "n_chains_per_rank": sampler_dict.get("n_chains_per_rank", ""),
            "n_samples_per_chain": sampler_dict.get("n_samples_per_chain", ""),
            "chunk_vstate": sampler_dict.get("chunk_vstate", ""),
        }
        table_sampler = _group_params(
            sampler_info, n_cols=2, max_str_length=max_str_lenght
        )

        # Información del schedule (training)
        schedule_dict = sim_dict.get("schedule", {})
        schedule_info = {}
        for k, v in schedule_dict.items():
            if k != "lr_schedule":  # Omitir lr_schedule completo
                if isinstance(v, dict):
                    # Para setup, tomar solo las primeras claves
                    if k == "setup":
                        schedule_info.update(
                            {
                                f"{k}.{sub_k}": sub_v
                                for sub_k, sub_v in list(v.items())[:2]
                            }
                        )
                    else:
                        schedule_info[k] = v
                else:
                    schedule_info[k] = v

        # Añadir info del lr_schedule si existe
        lr_schedule = schedule_dict.get("lr_schedule", {})
        if lr_schedule:
            lr_name = lr_schedule.get("name", "")
            schedule_info["lr_schedule"] = lr_name

        table_schedule = _group_params(
            schedule_info, n_cols=2, max_str_length=max_str_lenght
        )

        # Crear una tabla con dos columnas para mostrar lado a lado
        sim_layout = Table(show_header=False, box=None, pad_edge=False)
        sim_layout.add_column(width=40)
        sim_layout.add_column(width=40)

        sim_layout.add_row(
            Panel(table_sampler, title="[bold cyan]Sampler[/]", border_style=sim_color),
            Panel(
                table_schedule, title="[bold cyan]Schedule[/]", border_style=sim_color
            ),
        )

        panel_sim = Panel(
            sim_layout, title="[bold]SIMULATION[/]", border_style=sim_color
        )
        console.print(panel_sim)

    # ============================================================
    # COUPLING MODEL (sin "name" ni "size")
    # ============================================================
    cm_dict = settings.get("CM", {})
    if cm_dict:
        exclude_cm = {"name", "size", "model_name"}
        cm_params = {k: v for k, v in cm_dict.items() if k not in exclude_cm}
        table_cm = _group_params(cm_params, n_cols=3, max_str_length=max_str_lenght)
        panel_cm = Panel(
            table_cm,
            title=f"[bold]COUPLING MODEL ({cm_name})[/]",
            border_style=cm_color,
        )
        console.print(panel_cm)

    # ============================================================
    # NEURAL NETWORK (imprime el diccionario completo de forma legible)
    # ============================================================
    nn_dict = settings.get("NN", {})
    table_nn = _group_params(nn_dict, n_cols=2)
    panel_nn = Panel(
        table_nn, title=f"[bold]NEURAL NETWORK ({nn_name})[/]", border_style=nn_color
    )

    console.print(panel_nn)
    # if nn_dict:
    #     display_nn_architecture(console, nn_dict, __all_factories__, __all_factories__)

    console.rule("[bold green]")


def get_filenames_from_settings(cm_setup, nn_setup=None, sim_uuid=None, **kwargs):

    cm_name, nn_name = cm_setup["name"], nn_setup["name"]

    model_label = cm_name + "_" + nn_name
    size = cm_setup["size"]

    if cm_name == "Oxalate":
        strength = cm_setup["params"]["couplings"][0]
        theta = cm_setup["params"]["couplings"][1]
        phi = cm_setup["params"]["couplings"][2]
        cparams = f"_strength_{strength}_theta_{theta}_phi_{phi}"
        call_params = r"$a = %.1f$  $\theta = %.1f$  $\phi = %.1f$" % (
            strength,
            theta,
            phi,
        )

    elif cm_name == "IsingSquare":
        fields = cm_setup["params"]["fields"]
        field_ops = cm_setup["params"]["ops"][0]
        couplings = cm_setup["params"]["J"]
        coupling_ops = cm_setup["params"]["ops"][1]

        flat_fields = ""
        field_values = ""
        call_params = ""
        for f, v in zip(field_ops, fields):
            flat_fields += f + "_"
            field_values += f"{v}" + "_"
            call_params += f"{f}:{v}  "

        flat_couplings = ""
        coupling_values = ""

        # Just 1 coupling for Ising
        flat_couplings += coupling_ops[0] + "_"
        coupling_values += f"{couplings}" + "_"
        call_params += f"{coupling_ops[0]}:{couplings}  "

        cparams = f"_{flat_fields}{field_values}_{flat_couplings}{coupling_values}"

    elif cm_name == "Chain_YYZZ":
        fields = cm_setup["params"]["fields"]
        couplings = cm_setup["params"]["couplings"]
        field_ops = cm_setup["params"]["ops"][0]
        coupling_ops = cm_setup["params"]["ops"][1]
        flat_fields = ""
        field_values = ""
        call_params = ""
        for f, v in zip(field_ops, fields):
            flat_fields += f + "_"
            field_values += f"{v}" + "_"
            call_params += f"{f}:{v}  "

        flat_couplings = ""
        coupling_values = ""
        for f, v in zip(coupling_ops, couplings):
            flat_couplings += f + "_"
            coupling_values += f"{v}" + "_"
            call_params += f"{f}:{v}  "

        cparams = f"_{flat_fields}{field_values}_{flat_couplings}{coupling_values}"

    elif cm_name in ["LRChain", "LRSquare"]:
        J = cm_setup["params"]["J"]
        alpha = cm_setup["params"]["alpha"]
        fields = cm_setup["params"]["fields"]
        cparams = f"_J_{J}_alpha_{alpha}_XZ_{fields[0]}_{fields[1]}"
        call_params = r"$J=%.2f$  $\alpha=%.1f$  $XZ=(%.1f,%.1f)$" % (
            J,
            alpha,
            fields[0],
            fields[1],
        )

    elif cm_name == "J1J2Square":
        J1 = cm_setup["params"]["J1"]
        J2 = cm_setup["params"]["J2"]
        fields = cm_setup["params"]["fields"]
        cparams = f"_J1J2_{J1}_{J2}_XYZ_{fields[0]}_{fields[1]}_{fields[2]}"
        call_params = r"$J1=%.2f$  $J2=%.2f$  $XYZ=(%.1f,%1.f,%.1f)$" % (
            J1,
            J2,
            fields[0],
            fields[1],
            fields[2],
        )

    # Necessary to set unique simulation labels
    date = datetime.now().strftime("%Y%m%dT%H%M%S")

    if nn_setup["setup"]:
        setup = nn_setup["setup"]
        last_stage = next(reversed(setup))
        nnparams = get_string_from_nnsetup(setup[last_stage]["template"])
    else:
        nnparams = ""

    sim_label = (
        model_label
        + f"_simulation_{size[0]}x{size[1]}"
        + cparams
        + nnparams
        + f"_date_{date}_UUID_{sim_uuid}"
    )
    ED_label = (
        model_label
        + f"_xED_{size[0]}x{size[1]}"
        + cparams
        + f"_date_{date}_UUID_{sim_uuid}"
    )
    json_label = (
        model_label
        + f"_results_{size[0]}x{size[1]}"
        + cparams
        + nnparams
        + f"_date_{date}_UUID_{sim_uuid}"
    )
    title_label_callback = (
        f"Callback  " + model_label + "  " + call_params + f"  ({size[0]}x{size[1]})"
    )

    return sim_label, ED_label, json_label, title_label_callback
