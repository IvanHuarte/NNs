import uuid
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

import NN_module.models
from NN_module.models import __all_single__


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

    cleaned["SIM"]["split_training"] = split_training
    cleaned["SIM"]["sampler"] = sim["sampler"]
    cleaned["SIM"]["schedule"] = {}
    cleaned["SIM"]["schedule"]["split_training"] = True
    cleaned["SIM"]["schedule"]["setup"] = training_setup

    cleaned["SIM"]["schedule"]["lr_schedule"] = {}
    cleaned["SIM"]["schedule"]["lr_schedule"]["name"] = lr_name
    cleaned["SIM"]["schedule"]["lr_schedule"]["setup"] = lr_schedule_setup

    # Coupling model
    cleaned["CM"] = {}

    cm_name = cm["name"]
    cm_setup = cm["setup"]

    cleaned["CM"]["name"] = cm_name
    for k, v in cm_setup.items():
        if not "_list" in k:
            cleaned["CM"][k] = v

    cleaned["CM"]["size"] = cm_setup["size"]

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


def display_simulation_settings(settings, n_cols=5):
    console = Console()
    cm_sel = settings["CM"]["selection"]
    nn_sel = settings["NN"]["selection"]

    cm_color = "red"
    nn_color = "bright_yellow"

    # Título principal
    title_text = f"[bold blue]🧲 CM:[/] [{cm_color}]{cm_sel}[/]   [bold blue]🧠 NN_architecture:[/] [{nn_color}]{nn_sel}[/]"
    console.rule(title_text)

    def _group_params(params_dict):
        # Filtra claves no relevantes
        items = [(k, v) for k, v in params_dict.items() if not k.endswith("_list")]
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

    # Size
    size = settings.get("size", None)
    if isinstance(size, (list, tuple)) and all(isinstance(x, int) for x in size):
        size_str = "x".join(map(str, size))
        console.print(f"[bold yellow]🧱 Size:[/] [cyan]{size_str}[/]\n")

    # Panel de acoplamiento
    cm_dict = settings["CM"].get(cm_sel, {})
    if isinstance(cm_dict, dict):
        table_cm = _group_params(cm_dict)
        panel_cm = Panel(table_cm, title=f"[bold]{cm_sel}[/]", border_style=cm_color)
        console.print(panel_cm)

    # Panel de red neuronal
    nn_dict = settings["NN"].get(nn_sel, {})
    if isinstance(nn_dict, dict):
        table_nn = _group_params(nn_dict)
        panel_nn = Panel(table_nn, title=f"[bold]{nn_sel}[/]", border_style=nn_color)
        console.print(panel_nn)

    console.rule("[bold green]")


def architecture_label(name, model_setup):

    if name == "MLP":
        set_dim = [f"D({dim})" for dim in model_setup["hidden_alpha"]]
        set_act = [f"A({act})" for act in model_setup["activation"]]
        setup = "||"
        for i in range(len(model_setup["hidden_alpha"])):
            setup += f" {set_dim[i]} |"
            setup += f" {set_act[i]} |" if "0" not in set_act[i] else ""
        setup += "|"
        architecture = setup

    elif name in ["ViT", "ViT_2D"]:
        architecture = f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"

    elif name == "CvT":
        architecture = f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}   CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}   final_arch: {model_setup['final_architecture']} ||\n"
        architecture += f"|| symm_Z2: {model_setup['symm_Z2']}  trivial: {model_setup['trivial_Z2']} ||\n"

    elif name == "CvT2":
        architecture = f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}   CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  strides:{model_setup['strides']}  kernel: {model_setup['kernel']}   ||\n"
        architecture += f"|| final_arch: {model_setup['final_architecture']}  symm_Z2: {model_setup['symm_Z2']}  trivial: {model_setup['trivial_Z2']} ||\n"

    elif name == "CvT3":
        architecture = f"|| symm_Z2: {model_setup['symm_Z2']}   trivial: {model_setup['trivial_Z2']} ||"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}  CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}  final_arch: {model_setup['final_architecture']} ||\n"
        architecture += f"|| 2heads: {model_setup['two_heads']}  2heads_SC: {model_setup['two_heads_sincos']}  phasors: :{model_setup['phasors']} ||\n"

    elif name == "CvTaps":
        architecture = f"||        symm_Z2: {model_setup['symm_Z2']}   trivial: {model_setup['trivial_Z2']}             ||"
        architecture += f"|| blocks: {model_setup['n_CP_blocks']}  channels: {model_setup['channels']}  strides: {model_setup['strides']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}  final_arch: {model_setup['final_architecture']} ||\n"
        architecture += f"|| 2heads: {model_setup['two_heads']}  2heads_SC: {model_setup['two_heads_sincos']}  phasors: :{model_setup['phasors']} ||\n"

    elif name == "SplitTraining_ViT_MLP":
        architecture = f"ViT \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"MLP \n"
        set_dim = [f"D({dim})" for dim in model_setup["hidden_alpha"]]
        set_act = [f"A({act})" for act in model_setup["activation"]]
        setup = "||"
        for i in range(len(model_setup["hidden_alpha"])):
            setup += f" {set_dim[i]} |"
            setup += f" {set_act[i]} |" if "0" not in set_act[i] else ""
        setup += "|"
        architecture += setup

    elif name == "ViT2D_CNN":
        architecture = f" 2D: {model_setup['symm_2D']} || Z2: {model_setup['symm_Z2']} || trivial: {model_setup['trivial_Z2']} \n"
        architecture += f"ViT 2D \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"CNN\n"
        architecture += f"|| block_channels: {model_setup['block_channels']}    kernel:{model_setup['kernel_size']}    n_ffn_lay:{model_setup['n_ffn_layers_cnn']} ||"

    elif name == "ViT2D_CNNClsf":
        architecture = f" 2D: {model_setup['symm_2D']} || Z2: {model_setup['symm_Z2']} || trivial: {model_setup['trivial_Z2']} \n"
        architecture += f"ViT 2D \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"CNNClsf\n"
        architecture += f"||       channels:  {model_setup['cnnclsf_channels']}    n_classes: {model_setup['n_classes']}      ||"

    elif name == "SplitTraining_CvT_CNN":
        architecture = f"CvT \n"
        architecture += f"|| b: {model_setup['lattice_size']}  D_emb: {model_setup['CTemb_channels']}  heads: {model_setup['attn_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_CP_blocks']}   CP_channels: {model_setup['CP_channels']} ||\n"
        architecture += f"|| kernel: {model_setup['kernel']}   final_architecture: {model_setup['final_architecture']} ||\n"
        architecture += f"CNN\n"
        architecture += f"|| block_channels: {model_setup['block_channels_cnn']}    kernel:{model_setup['kernel_size_cnn']}    n_ffn_lay:{model_setup['n_ffn_layers_cnn']} ||"

    elif name == "SplitTraining_CvT_CvT":
        architecture = f"CvT 1\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels_1']}  n_CPB: {model_setup['n_CP_blocks_1']}   CP_ch: {model_setup['CP_channels_1']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_1']}  kernel: {model_setup['kernel_1']}   final_arch: {model_setup['final_architecture_1']} ||\n"
        architecture += f"CvT 2\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels_2']}  n_CPB: {model_setup['n_CP_blocks_2']}   CP_ch: {model_setup['CP_channels_2']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_2']}  kernel: {model_setup['kernel_2']}   final_arch: {model_setup['final_architecture_2']} ||\n"

    elif name == "CvT3_CvT3":
        architecture = f"CvT3 1\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels_1']}  n_CPB: {model_setup['n_CP_blocks_1']}   CP_ch: {model_setup['CP_channels_1']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_1']}  kernel: {model_setup['kernel_1']}   final_arch: {model_setup['final_architecture_1']} ||\n"
        architecture += f"CvT3 2\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels_2']}  n_CPB: {model_setup['n_CP_blocks_2']}   CP_ch: {model_setup['CP_channels_2']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_2']}  kernel: {model_setup['kernel_2']}   final_arch: {model_setup['final_architecture_2']} ||\n"
        architecture += f"|| phasors:{model_setup['phasors']}  ||"
        architecture += f"            Z2: {model_setup['symm_Z2']} trivial: {model_setup['trivial_Z2']}         \n"

    elif name == "CvTaps_CvTaps":
        architecture = f"||        symm_Z2: {model_setup['symm_Z2']}   trivial: {model_setup['trivial_Z2']}             ||"
        architecture += f"CvTaps 1\n"
        architecture += f"|| blocks: {model_setup['n_CP_blocks_1']}  channels: {model_setup['channels_1']}  strides: {model_setup['strides_1']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_1']}  kernel: {model_setup['kernel_1']}  final_arch: {model_setup['final_architecture_1']} ||\n"
        architecture += f"CvTaps 2\n"
        architecture += f"|| blocks: {model_setup['n_CP_blocks_2']}  channels: {model_setup['channels_2']}  strides: {model_setup['strides_2']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_2']}  kernel: {model_setup['kernel_2']}  final_arch: {model_setup['final_architecture_2']} ||\n"
        architecture += f"||                        phasors: :{model_setup['phasors']}                        ||\n"

    elif name == "CvT3_CNNPh":
        architecture = f"CvT3\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}  CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}  final_arch: {model_setup['final_architecture']} ||\n"
        architecture += f"CNNPhasor\n"
        architecture += f"||       channels:  {model_setup['cnnph_channels']}        ||"
        architecture += f"||             symm_Z2: {model_setup['symm_Z2']}  trivial: {model_setup['trivial_Z2']}              ||"

    elif name == "CvT3_EDPPh":
        architecture = f"CvT3\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}  CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}  final_arch: {model_setup['final_architecture']} ||\n"
        architecture += f"EDPPh\n"
        architecture += f"||       channels:  {model_setup['edpph_channels']}        ||"
        architecture += f"||             Z2: {model_setup['symm_Z2']}  trivial: {model_setup['trivial_Z2']}              ||"
    elif name == "CvT3_CNNClsf":
        architecture = (
            f" Z2: {model_setup['symm_Z2']} || trivial: {model_setup['trivial_Z2']} \n"
        )
        architecture = f"CvT3\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}  CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}  final_arch: {model_setup['final_architecture']} ||\n"
        architecture += f"CNNClsf\n"
        architecture += f"||       channels:  {model_setup['cnnclsf_channels']}    n_classes: {model_setup['n_classes']}      ||"

    return architecture


def get_filenames_from_settings(cm_name, nn_name, sim_uuid, **kwargs):

    model_label = cm_name + "_" + nn_name
    size = kwargs["size"]

    if cm_name == "Oxalate":
        strength = kwargs["strength"]
        theta = kwargs["theta"]
        phi = kwargs["phi"]
        cparams = f"_strength_{strength}_theta_{theta}_phi_{phi}"
        call_params = r"$a = %.1f$  $\theta = %.1f$  $\phi = %.1f$" % (
            strength,
            theta,
            phi,
        )

    elif cm_name == "Chain_YYZZ":
        fields = kwargs["fields"]
        couplings = kwargs["couplings"]
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
        J = kwargs["J"]
        alpha = kwargs["alpha"]
        fields = kwargs["fields"]
        cparams = f"_J_{J}_alpha_{alpha}_XZ_{fields[0]}_{fields[1]}"
        call_params = r"$J=%.2f$  $\alpha=%.1f$  $XZ=(%.1f,%.1f)$" % (
            J,
            alpha,
            fields[0],
            fields[1],
        )

    elif cm_name == "J1J2Square":
        J1 = kwargs["J1"]
        J2 = kwargs["J2"]
        fields = kwargs["fields"]
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
    nnparams = get_string_from_nnsetup(kwargs) + f"_date_{date}_UUID_{sim_uuid}"

    sim_label = model_label + f"_simulation_{size[0]}x{size[1]}" + cparams + nnparams
    ED_label = model_label + f"_xED_{size[0]}x{size[1]}" + cparams
    json_label = model_label + f"_results_{size[0]}x{size[1]}" + cparams + nnparams
    title_label_callback = (
        f"Callback  " + model_label + "  " + call_params + f"  ({size[0]}x{size[1]})"
    )

    return sim_label, ED_label, json_label, title_label_callback


def get_ST_folder(split_training, setup):

    if split_training:
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
