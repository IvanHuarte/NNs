import uuid
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

import NN_module.models
from NN_module.models import __all_single__, __all_factories__


def get_sim_config(configurations, **kwargs):

    sim = configurations["SIM"]
    cm = configurations["CM"]
    nn = configurations["NN"]
    cleaned = {**kwargs}

    # Simulation
    cleaned["SIM"] = {}

    split_training = sim["split_training"]
    training_name = "ST_schedule" if split_training else "normal_schedule"
    lr_name = sim[training_name]["lr_name"]
    training_setup = sim[training_name]["setup"]
    lr_schedule_setup = sim[training_name]["lr_schedules"][lr_name]

    cleaned["SIM"]["sampler"] = sim["sampler"]
    cleaned["SIM"]["split_training"] = split_training
    cleaned["SIM"]["schedule"] = {}
    cleaned["SIM"]["schedule"]["split_training"] = True
    cleaned["SIM"]["schedule"]["setup"] = training_setup

    cleaned["SIM"]["schedule"]["lr_schedule"] = {}
    cleaned["SIM"]["schedule"]["lr_schedule"]["name"] = lr_name
    cleaned["SIM"]["schedule"]["lr_schedule"]["setup"] = lr_schedule_setup

    # Coupling model
    cleaned["CM"] = cm

    # NN architecture

    cleaned["NN"] = {}
    cleaned["NN"]["name"] = nn["name"]
    cleaned["NN"]["setup"] = nn["setup"]

    return cleaned


def get_string_from_nnsetup(dic, nivel=0):
    if not isinstance(dic, dict):
        return ""

    modulo = dic.get("module")
    setup = dic.get("setup")  # setup puede no existir en algunos niveles

    if modulo is None:
        return ""

    if modulo in __all_single__:
        return modulo

    nombres_submodulos = []

    # Si existe la clave 'setup' y es dict, se usa
    if isinstance(setup, dict):
        sub_setup = setup.get("setup")
        if isinstance(sub_setup, dict):
            for key, val in sub_setup.items():
                res = get_string_from_nnsetup(val, nivel + 1)
                if res:
                    nombres_submodulos.append(res)
        else:
            # Si no existe la clave 'setup' dentro, probamos iterar setup directamente
            for key, val in setup.items():
                # Evitar iterar claves no módulos, como flags booleanos, etc.
                if isinstance(val, dict):
                    res = get_string_from_nnsetup(val, nivel + 1)
                    if res:
                        nombres_submodulos.append(res)

    # En caso de que 'setup' no exista, intentar ver si el dicc contiene submódulos directos
    elif isinstance(dic, dict):
        # Iterar claves que no sean 'module' ni 'setup' ni otras claves conocidas no módulo
        for key, val in dic.items():
            if key not in ("module", "setup") and isinstance(val, dict):
                res = get_string_from_nnsetup(val, nivel + 1)
                if res:
                    nombres_submodulos.append(res)

    separador = "_" * (nivel + 1)
    if nombres_submodulos:
        hijos = separador.join(nombres_submodulos)
        return f"{modulo}{separador}{hijos}"
    else:
        return modulo


def get_write_folder_from_model(config):

    name = config["NN"]["selection"]
    nn_setup = config["NN"][name]
    model_label = config["CM"]["selection"]

    nn_label = get_string_from_nnsetup(nn_setup)
    nn_label = name + "_" + nn_label
    return config["write_folder_sim"] + model_label + "/" + nn_label + "/"


from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.box import ROUNDED


def display_nn_architecture(console, nn_dict, all_singles, all_factories):
    """
    Dibuja la arquitectura de la red neuronal como diagrama de flujo.

    Args:
        nn_dict: Diccionario con la configuración de la red neuronal
        all_singles: Lista/set de módulos simples (con parámetros entrenables)
        all_factories: Lista/set de módulos de flujo (Sequential, SplitTraining, Transversal)
    """

    # Obtener el root (sin "name")
    root_setup = nn_dict.get("setup", {})

    def _format_params(params_dict, exclude_keys=None):
        """Formatea parámetros para mostrar en un cuadro"""
        if exclude_keys is None:
            exclude_keys = {"module", "setup"}

        lines = []
        for k, v in params_dict.items():
            if k not in exclude_keys:
                v_str = str(v)
                if len(v_str) > 40:
                    v_str = v_str[:37] + "..."
                lines.append(f"[cyan]{k}[/]=[yellow]{v_str}[/]")

        return "\n".join(lines) if lines else "[dim]no params[/]"

    def _is_flow_module(module_name):
        """Verifica si es un módulo de flujo"""
        return module_name in all_factories

    def _is_simple_module(module_name):
        """Verifica si es un módulo simple"""
        return module_name in all_singles

    def _draw_module(module_dict, level=0, parent_name=""):
        module_name = module_dict.get("module", "Unknown")
        setup = module_dict.get("setup", {})

        extra_params = {
            k: v for k, v in module_dict.items() if k not in {"module", "setup"}
        }
        module_title = f"[bold magenta]{module_name}[/]"
        param_text = _format_params(extra_params, exclude_keys=set())

        module_panel = Panel(
            param_text,
            title=module_title,
            border_style="magenta",
            box=ROUNDED,
            width=30,
        )

        result = {"panel": module_panel, "name": module_name, "children": []}

        # Nodo terminal si no hay setup
        if not setup:
            return result

        if _is_flow_module(module_name):
            if module_name == "SplitTraining":
                branches = {}
                for key, value in setup.items():
                    if key.endswith("_setup") and isinstance(value, dict):
                        branch_name = key.replace("_setup", "")
                        branches[branch_name] = value
                for branch_name, branch_dict in branches.items():
                    child = _draw_module(branch_dict, level + 1, branch_name)
                    result["children"].append({"name": branch_name, "module": child})
            elif module_name == "Sequential":
                seq_modules = []
                for key, value in sorted(setup.items()):
                    if isinstance(value, dict) and "module" in value:
                        seq_modules.append((key, value))
                for idx, (key, seq_dict) in enumerate(seq_modules):
                    child = _draw_module(seq_dict, level + 1, f"Seq_{idx}")
                    result["children"].append(
                        {"name": f"Seq_{idx}", "module": child, "sequential": True}
                    )
            elif module_name == "Transversal":
                trans_modules = []
                for key, value in sorted(setup.items()):
                    if isinstance(value, dict) and "module" in value:
                        trans_modules.append((key, value))
                for idx, (key, trans_dict) in enumerate(trans_modules):
                    child = _draw_module(trans_dict, level + 1, f"Trans_{idx}")
                    result["children"].append({"name": f"Trans_{idx}", "module": child})
        elif _is_simple_module(module_name):
            pass  # módulos simples no tienen hijos de flujo

        return result

    def _render_branches(parents, children_names, children_panels):
        """Renderiza branches bien alineadas para N hijos bajo el padre"""
        col_count = len(children_panels)
        table = Table.grid(padding=(0, 1))
        for _ in range(col_count):
            table.add_column(justify="center")

        # Espacio arriba
        table.add_row(*([""] * col_count))
        # Padre centrado
        mid = col_count // 2
        parent_row = [""] * col_count
        parent_row[mid] = parents
        table.add_row(*parent_row)
        # Línea descendente
        branch_row = [""] * col_count
        for i in range(col_count):
            branch_row[i] = Text("│", style="dim") if i == mid else ""
        table.add_row(*branch_row)
        # Bifurcación horizontal
        branch_row = [""] * col_count
        for i in range(col_count):
            branch_row[i] = Text("└──", style="dim") if i == mid else ""
        table.add_row(*branch_row)
        # Hijos
        table.add_row(*children_panels)
        return table

    def _render_tree(tree_node, is_root=False):
        renderables = []
        renderables.append(tree_node["panel"])

        if tree_node["children"]:
            children = tree_node["children"]
            is_sequential = len(children) > 0 and children[0].get("sequential", False)

            if is_sequential:
                renderables.append(Text("    │", style="dim"))
                renderables.append(Text("    ↓", style="dim"))
                for child in children:
                    child_renderables = _render_tree(child["module"])
                    renderables.extend(child_renderables)
                    if child != children[-1]:
                        renderables.append(Text("    │", style="dim"))
                        renderables.append(Text("    ↓", style="dim"))
            else:
                # Mejor alineación para ramas paralelas
                child_panels = []
                child_names = []
                for child in children:
                    child_tree_panel = _render_tree(child["module"])
                    group = Table.grid()
                    group.add_column()
                    for item in child_tree_panel:
                        group.add_row(item)
                    child_panels.append(group)
                    child_names.append(child["name"])
                # Llama a _render_branches para organizar padre e hijos
                renderables.append(
                    _render_branches(tree_node["panel"], child_names, child_panels)
                )

        return renderables

    # Construir el árbol desde el root
    tree = _draw_module(root_setup)

    # Renderizar el árbol
    rendered = _render_tree(tree, is_root=True)

    # Mostrar todo en un panel
    main_table = Table.grid()
    main_table.add_column(justify="center")

    for item in rendered:
        main_table.add_row(item)

    nn_name = nn_dict.get("name", "Unknown")
    final_panel = Panel(
        main_table,
        title=f"[bold yellow]NEURAL NETWORK: {nn_name}[/]",
        border_style="bright_yellow",
        expand=False,
    )

    console.print(final_panel)


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


def get_filenames_from_settings(cm_setup, nn_setup, sim_uuid=None, **kwargs):

    cm_name, nn_name = cm_setup["name"], nn_setup["name"]

    model_label = cm_name + "_" + nn_name
    size = cm_setup["size"]

    if cm_name == "Oxalate":
        strength = cm_setup["params"]["strength"]
        theta = cm_setup["params"]["theta"]
        phi = cm_setup["params"]["phi"]
        cparams = f"_strength_{strength}_theta_{theta}_phi_{phi}"
        call_params = r"$a = %.1f$  $\theta = %.1f$  $\phi = %.1f$" % (
            strength,
            theta,
            phi,
        )

    elif cm_name == "Chain_YYZZ":
        fields = cm_setup["params"]["fields"]
        couplings = cm_setup["params"]["couplings"]
        flat_fields = ""
        field_values = ""
        call_params = ""
        for f, v in zip(["X", "Y", "Z"], fields):
            flat_fields += f + "_"
            field_values += f"{v}" + "_"
            call_params += f"{f}:{v}  "

        flat_couplings = ""
        coupling_values = ""
        for f, v in zip(["XX", "YY", "ZZ"], couplings):
            flat_couplings += f + "_"
            coupling_values += f"{v}" + "_"
            call_params += f"{f}:{v}  "

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
    nnparams = (
        get_string_from_nnsetup(nn_setup["setup"]) + f"_date_{date}_UUID_{sim_uuid}"
    )

    sim_label = model_label + f"_simulation_{size[0]}x{size[1]}" + cparams + nnparams
    ED_label = model_label + f"_xED_{size[0]}x{size[1]}" + cparams
    json_label = model_label + f"_results_{size[0]}x{size[1]}" + cparams + nnparams
    title_label_callback = (
        f"Callback  " + model_label + "  " + call_params + f"  ({size[0]}x{size[1]})"
    )

    return sim_label, ED_label, json_label, title_label_callback


def get_ST_folder(split_training, setup):

    if split_training:
        setup = setup["setup"]
        label = "ST"

        s, m, r = setup.values()
        for seg, mode, repeats in zip(s, m, r):
            label += "_"
            for s, m in zip(seg, mode):
                label += f"{s}{m}"
            label += f"x{repeats}"

    else:
        label = "Both"

    return label
