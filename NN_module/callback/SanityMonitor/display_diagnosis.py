from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

import shutil
from NN_module.utils import print_tree

THEME = {
    # Colores de bordes
    "panel_border": "steel_blue3",
    "gradient_border": "medium_purple3",
    # Títulos
    "title_color": "deep_sky_blue1",
    "subtitle_color": "light_sky_blue3",
    # Texto
    "text_primary": "white",
    "text_secondary": "light_steel_blue3",
    "text_dim": "gray50",
    # Estados
    "ok": "#10921B",  # verde oscuro
    "warning": "#B8A51A",  # amarillo/ocre
    "error": "#B22222",  # rojo fuego
}


def display_phase(data, verbose, message_counter):

    table = Table(box=None)

    table.add_column("", style="red", justify="center")
    table.add_column(
        "Mean (|⟨e^(q i·ϕ)⟩|)",
        style="bold red",
        header_style="#8B1A1A",
        highlight=True,
        justify="center",
    )
    table.add_column(
        "Std (σ[e^(q i·ϕ)])",
        style="red",
        header_style="#8B1A1A",
        highlight=True,
        justify="center",
    )
    table.add_column(
        "status (mean/std)",
        style="red",
        header_style=THEME["subtitle_color"],
        justify="center",
    )
    if verbose == 4:
        table.add_column("", style=THEME["text_dim"], justify="center")

    for q_name, q_data in data.items():
        mean = q_data["mean"]
        std = q_data["std"]
        if verbose == 4:
            table.add_row(
                q_name,
                mean["label"],
                std["label"],
                f"{mean['status']}/{std['status']}",
                f"{message_counter}/{message_counter+1}",
            )
            message_counter += 2
        else:
            table.add_row(
                q_name,
                mean["label"],
                std["label"],
                f"{mean['status']}/{std['status']}",
                "",
            )
    return table, message_counter


def display_samples_histogram(data):

    data = data["histogram"]["label"]
    table = Table(box=None)

    table.add_column("Magnetization", style="red", justify="center")
    table.add_column(
        "Number of samples",
        style="bold red",
        header_style="#8B1A1A",
        highlight=True,
        justify="center",
    )

    magnetization, counts = (
        data["magnetization"],
        data["counts"],
    )

    for m, count in zip(magnetization, counts):
        table.add_row(
            f"{m}",
            f"{count}",
        )
    return table


def display_diagnosis_sanity_monitor(diagnosis: dict, verbose: int = 3) -> None:
    """
    Displays a beautiful, organized sanity monitor for VMC quantum simulation diagnostics.

    The display includes:
    - A prominent title banner
    - Overall status summary
    - Specially formatted GRADIENTS section (2-column layout: analysis + distribution)
    - Other metrics in 2-column grid layout
    - Optional messages section with indexed references

    Args:
        diagnosis (dict): Dictionary containing diagnostic metrics with structure:
            {
                'gradients': {
                    'norm_per_sample': {'mean': {...}, 'std': {...}},
                    'global_norm': {...},
                    'ReIm_norms': {'global': {...}, 'ModulusNet': {...}, 'PhaseNet': {...}},
                    'percentage_norm_intervals': {...}
                },
                'sampling': {...},
                'phase': {...},
                'overall': {'status': '🔴', 'message': '...'}
            }

        verbose (int): Control output verbosity (default: 3)
                      - 0: No output
                      - 1: Only ["❓", "🔴"] status
                      - 2: Only ["🔴", "🟡", "⚠️", "⚠️⚠️"] status (warnings + errors)
                      - 3: All status symbols (default)
                      - 4: All status + messages section with indexed references

    Returns:
        None (prints to console via Rich)

    Example:
        >>> diagnosis = {
        ...     'gradients': {
        ...         'norm_per_sample': {
        ...             'mean': {
        ...                 'label': '⟨||∇logψ||⟩_s = 7.83e-03',
        ...                 'status': '✅',
        ...                 'message': 'SANO. Gradientes normales → Entrenando bien'
        ...             },
        ...             'std': {...}
        ...         },
        ...         'global_norm': {...},
        ...         'ReIm_norms': {...},
        ...         'percentage_norm_intervals': {...}
        ...     },
        ...     'sampling': {...},
        ...     'phase': {...},
        ...     'overall': {'status': '🔴', 'message': 'Intervenir'}
        ... }
        >>> display_diagnosis_sanity_monitor(diagnosis, verbose=4)
    """

    if verbose == 0:
        return

    terminal_size = shutil.get_terminal_size()
    width = terminal_size.columns

    console = Console(
        width=width,
        force_terminal=True,
    )

    # ==================== FILTERING FUNCTION ====================
    def should_display_status(status: str) -> bool:
        """Determine if a status should be displayed based on verbose level"""
        if verbose == 3 or verbose == 4:
            return True
        elif verbose == 1:
            return status in ["❓", "🔴"]
        elif verbose == 2:
            return status in ["🔴", "🟡", "⚠️", "⚠️⚠️"]
        return False

    # ==================== COLOR FUNCTION FOR INTERVALS ====================
    def get_interval_color(interval_str: str) -> str:
        """Map interval strings to colors for gradient norm distribution"""
        color_map = {
            "< 10⁻⁴": "red",  # Extremo
            "> 10³": "red",  # Extremo
            "10⁻⁴ <--> 10⁻³": "bright_red",  # Peligroso
            "10² <--> 10³": "bright_red",  # Peligroso
            "10⁻³ <--> 10⁻²": "yellow",  # Medio peligroso
            "10¹ <--> 10²": "yellow",  # Medio peligroso
            "10⁻² <--> 10⁻¹": "green",  # Bueno
            "10⁻¹ <--> 10⁰": "bright_green",  # Bueno
            "10⁰ <--> 10¹": "green",  # Bueno
        }
        return color_map.get(interval_str, THEME["text_primary"])

    # ==================== TITLE ====================
    console.print("\n")
    console.print(
        Panel(
            Text("🔬 SANITY MONITOR", justify="center", style=THEME["subtitle_color"]),
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # ==================== MESSAGE TRACKING ====================
    messages_dict = {}  # For verbose=4
    message_counter = 1

    # ==================== METRICS DISPLAY ====================
    console.print("\n")

    # Prepare metrics for grid (except 'overall')
    metrics_keys = [k for k in diagnosis.keys() if k != "overall"]

    # Separate 'gradients' from the rest
    has_gradients = "gradients" in metrics_keys
    other_metrics = [k for k in metrics_keys if k != "gradients"]

    # ==================== GRADIENTS (Special Treatment) ====================
    if has_gradients:
        gradients_data = diagnosis["gradients"]

        # Create left panel: global_norm, norm_per_sample, ReIm_norms
        left_content = Text(justify="center")
        left_content.append("Global Norm:   ", style=THEME["subtitle_color"])
        # left_content.append("─" * 35 + "\n", style="dim")

        # global_norm
        if "global_norm" in gradients_data:
            gn = gradients_data["global_norm"]
            left_content.append(gn["label"], style=THEME["text_primary"])
            if should_display_status(gn["status"]):
                if verbose == 4:
                    left_content.append(
                        f"  {gn['status']}", style=THEME["text_primary"]
                    )
                    messages_dict[message_counter] = gn["message"]
                    left_content.append(
                        f"   ({message_counter})\n", style=THEME["text_dim"]
                    )
                    message_counter += 1
                else:
                    left_content.append(
                        f"  {gn['status']}\n", style=THEME["text_primary"]
                    )

                left_content.append("\n")

        # norm_per_sample
        if "norm_per_sample" in gradients_data:
            nps = gradients_data["norm_per_sample"]
            left_content.append("Norm/sample:      ", style=THEME["subtitle_color"])

            if "mean" in nps:
                mean_data = nps["mean"]
                left_content.append(mean_data["label"], style=THEME["text_primary"])
                if should_display_status(mean_data["status"]):
                    if verbose == 4:
                        left_content.append(
                            f"  {mean_data['status']}", style=THEME["text_primary"]
                        )
                        messages_dict[message_counter] = mean_data["message"]
                        left_content.append(
                            f"   ({message_counter})\n", style=THEME["text_dim"]
                        )
                        message_counter += 1
                    else:
                        left_content.append(
                            f"  {mean_data['status']}\n", style=THEME["text_primary"]
                        )

                    left_content.append("\n")

            left_content.append("                              ")

            if "std" in nps:
                std_data = nps["std"]
                left_content.append(std_data["label"], style=THEME["text_primary"])
                if should_display_status(std_data["status"]):
                    if verbose == 4:
                        left_content.append(
                            f"  {std_data['status']}", style=THEME["text_primary"]
                        )
                        messages_dict[message_counter] = std_data["message"]
                        left_content.append(
                            f"   ({message_counter})\n", style=THEME["text_dim"]
                        )
                        message_counter += 1
                    else:
                        left_content.append(
                            f"  {std_data['status']}\n", style=THEME["text_primary"]
                        )

                    left_content.append("\n")

        # ReIm_norms
        if "ReIm_norms" in gradients_data:
            left_content.append("\nRe/Im norm ratios\n", style=THEME["title_color"])
            left_content.append("─" * 25 + "\n\n", style="dim")

            reim = gradients_data["ReIm_norms"]

            for key in ["global", "ModulusNet", "PhaseNet"]:

                if key in reim:
                    reim_data = reim[key]
                    left_content.append(
                        f"{key.capitalize()}:   ", style=THEME["subtitle_color"]
                    )
                    left_content.append(reim_data["label"], style=THEME["text_primary"])
                    if should_display_status(reim_data["status"]):
                        if verbose == 4:
                            left_content.append(
                                f"  {reim_data['status']}", style=THEME["text_primary"]
                            )
                            messages_dict[message_counter] = reim_data["message"]
                            left_content.append(
                                f"   ({message_counter})\n", style=THEME["text_dim"]
                            )
                            message_counter += 1
                        else:
                            left_content.append(
                                f"  {reim_data['status']}\n",
                                style=THEME["text_primary"],
                            )

                        left_content.append("\n")

        # ==================== PERCENTAGE INTERVALS (Right Side) ====================
        right_content = Text(justify="center")
        right_content.append("Gradient Norm Distribution\n", style=THEME["title_color"])
        right_content.append("─" * 35 + "\n\n", style="dim")

        if "percentage_norm_intervals" in gradients_data:
            pni = gradients_data["percentage_norm_intervals"]

            # Status and message for the metric itself
            if should_display_status(pni["status"]):
                right_content.append("Status: ", style=THEME["text_secondary"])
                if verbose == 4:
                    messages_dict[message_counter] = pni["message"]
                    right_content.append(
                        f"{pni['status']}", style=THEME["text_primary"]
                    )
                    right_content.append(
                        f"   ({message_counter})\n", style=THEME["text_dim"]
                    )
                    message_counter += 1
                else:
                    right_content.append(
                        f"{pni['status']}\n", style=THEME["text_primary"]
                    )

            right_content.append("\n")

            # Parse label for interval data
            label_text = pni.get("label", "")

            if isinstance(label_text, dict):
                for interval_name, percentage in label_text.items():
                    color = get_interval_color(interval_name)
                    right_content.append(
                        f"{interval_name}: ", style=THEME["text_primary"]
                    )
                    right_content.append(f"{percentage}\n", style=color)
            elif isinstance(label_text, list):
                intervals = [
                    "< 10⁻⁴",
                    "10⁻⁴ <--> 10⁻³",
                    "10⁻³ <--> 10⁻²",
                    "10⁻² <--> 10⁻¹",
                    "10⁻¹ <--> 10⁰",
                    "10⁰ <--> 10¹",
                    "10¹ <--> 10²",
                    "10² <--> 10³",
                    "> 10³",
                ][::-1]
                for i, percentage in enumerate(label_text):
                    interval = intervals[i] if i < len(intervals) else "Unknown"
                    color = get_interval_color(interval)
                    # right_content.append(f"{interval}: ", style=THEME['text_primary'])
                    right_content.append(f"{percentage}\n", style=color)
            else:
                right_content.append("Message: ", style="dim cyan")
                right_content.append(f"{label_text}\n", style=THEME["text_primary"])

        # Create panel with two columns
        gradient_panel_content = Columns(
            [
                Panel(
                    left_content,
                    title="[bold][/bold]",
                    border_style=THEME["text_primary"],
                    box=box.SIMPLE,
                    padding=(1, 2),
                ),
                Panel(
                    right_content,
                    title="[bold][/bold]",
                    border_style=THEME["text_primary"],
                    box=box.SIMPLE,
                    padding=(1, 2),
                ),
            ],
            equal=True,
            expand=True,
        )

        # Get status for gradients title
        gradient_status = (
            gradients_data.get("overall", {}).get("status", "🟢")
            if "overall" in gradients_data
            else "🟢"
        )

        console.print(
            Panel(
                gradient_panel_content,
                title=f"[bold]📊 GRADIENTS {gradient_status}[/bold]",
                border_style=THEME["panel_border"],
                padding=(1, 2),
                expand=True,
            )
        )

    # ==================== OTHER METRICS (2-column layout) ====================
    if other_metrics:
        console.print("\n")

        # Group metrics in pairs
        metric_pairs = []
        for i in range(0, len(other_metrics), 2):
            if i + 1 < len(other_metrics):
                metric_pairs.append((other_metrics[i], other_metrics[i + 1]))
            else:
                metric_pairs.append((other_metrics[i], None))

        # For each pair, create two columns
        for left_metric, right_metric in metric_pairs:

            # ========== CREATE LEFT PANEL ==========
            left_text = Text(justify="center")
            left_metric_data = diagnosis[left_metric]

            for submetric_name, submetric_data in left_metric_data.items():

                if left_metric == "phase":
                    left_text, message_counter = display_phase(
                        left_metric_data, verbose, message_counter
                    )
                    break

                if left_metric == "sample_histogram":
                    left_text = display_samples_histogram(left_metric_data)
                    break

                if isinstance(submetric_data, dict) and "status" in submetric_data:
                    left_text.append(
                        submetric_data["label"], style=THEME["text_primary"]
                    )
                    if should_display_status(submetric_data["status"]):
                        if verbose == 4:
                            messages_dict[message_counter] = submetric_data["message"]
                            left_text.append(
                                f"  {submetric_data['status']}",
                                style=THEME["text_primary"],
                            )
                            left_text.append(
                                f"  ({message_counter})\n", style=THEME["text_dim"]
                            )
                            message_counter += 1
                        else:
                            left_text.append(
                                f"  {submetric_data['status']}\n",
                                style=THEME["text_primary"],
                            )

                        left_text.append("\n")

                elif isinstance(submetric_data, dict):

                    left_text.append(
                        f"{submetric_name}:\n", style=THEME["subtitle_color"]
                    )

                    for (
                        subsubmetric_name,
                        subsubmetric_data,
                    ) in submetric_data.items():

                        if (
                            isinstance(subsubmetric_data, dict)
                            and "status" in subsubmetric_data
                        ):
                            left_text.append(
                                subsubmetric_data["label"], style=THEME["text_primary"]
                            )
                            if should_display_status(subsubmetric_data["status"]):
                                if verbose == 4:
                                    messages_dict[message_counter] = subsubmetric_data[
                                        "message"
                                    ]
                                    left_text.append(
                                        f"  {subsubmetric_data['status']}",
                                        style=THEME["text_primary"],
                                    )
                                    left_text.append(
                                        f"  ({message_counter})\n",
                                        style=THEME["text_dim"],
                                    )
                                    message_counter += 1
                                else:
                                    left_text.append(
                                        f"  {subsubmetric_data['status']}\n",
                                        style=THEME["text_primary"],
                                    )

                                left_text.append("\n")

                else:
                    raise Warning(
                        f"Incorrect data format for metric '{submetric_name}' in '{left_metric}'"
                    )

            left_panel = Panel(
                left_text,
                title=f"[bold]{left_metric.upper()}[/bold]",
                border_style=THEME["panel_border"],
                padding=(1, 2),
                expand=True,
            )

            # ========== CREATE RIGHT PANEL ==========
            if right_metric:
                right_text = Text(
                    justify="center",
                )
                right_metric_data = diagnosis[right_metric]
                for submetric_name, submetric_data in right_metric_data.items():

                    if right_metric == "phase":
                        right_text, message_counter = display_phase(
                            right_metric_data, verbose, message_counter
                        )
                        break

                    if right_metric == "sample_histogram":
                        right_text = display_samples_histogram(right_metric_data)
                        break

                    if isinstance(submetric_data, dict) and "status" in submetric_data:
                        right_text.append(
                            submetric_data["label"], style=THEME["text_primary"]
                        )
                        if should_display_status(submetric_data["status"]):
                            if verbose == 4:
                                messages_dict[message_counter] = submetric_data[
                                    "message"
                                ]
                                right_text.append(
                                    f"  {submetric_data['status']}",
                                    style=THEME["text_primary"],
                                )
                                right_text.append(
                                    f"  ({message_counter})\n", style=THEME["text_dim"]
                                )
                                message_counter += 1
                            else:
                                right_text.append(
                                    f"  {submetric_data['status']}\n",
                                    style=THEME["text_primary"],
                                )

                            right_text.append("\n")

                    elif isinstance(submetric_data, dict):

                        for (
                            subsubmetric_name,
                            subsubmetric_data,
                        ) in submetric_data.items():

                            right_text.append(
                                f"{subsubmetric_name}:\n", style=THEME["subtitle_color"]
                            )

                            if (
                                subsubmetric_name == "histogram"
                                and left_metric == "sampling"
                            ):
                                left_text.append("\n")
                                left_text.append(
                                    display_samples_histogram(subsubmetric_data),
                                    style=THEME["text_primary"],
                                )
                                left_text.append("\n")
                                continue

                            if (
                                isinstance(subsubmetric_data, dict)
                                and "status" in subsubmetric_data
                            ):
                                right_text.append(
                                    subsubmetric_data["label"],
                                    style=THEME["text_primary"],
                                )
                                if should_display_status(subsubmetric_data["status"]):
                                    if verbose == 4:
                                        messages_dict[message_counter] = (
                                            subsubmetric_data["message"]
                                        )
                                        right_text.append(
                                            f"  {subsubmetric_data['status']}",
                                            style=THEME["text_primary"],
                                        )
                                        right_text.append(
                                            f"  ({message_counter})\n",
                                            style=THEME["text_dim"],
                                        )
                                        message_counter += 1
                                    else:
                                        right_text.append(
                                            f"  {subsubmetric_data['status']}\n",
                                            style=THEME["text_primary"],
                                        )

                                    right_text.append("\n")

                    else:
                        raise Warning(
                            f"Incorrect data format for metric '{submetric_name}' in '{left_metric}'"
                        )

                right_panel = Panel(
                    right_text,
                    title=f"[bold]{right_metric.upper()}[/bold]",
                    border_style=THEME["panel_border"],
                    padding=(1, 2),
                    expand=True,
                )

                row = Columns([left_panel, right_panel], equal=True)
                console.print(row)
            else:
                console.print(left_panel)

            console.print()

    # ==================== MESSAGES SECTION (verbose=4) ====================
    if verbose == 4 and messages_dict:
        console.print("\n" + "─" * 80)
        messages_text = Text()
        messages_text.append("📋 MESSAGES\n", style=THEME["text_dim"])
        messages_text.append("─" * 80 + "\n\n", style="dim")

        for msg_num in sorted(messages_dict.keys()):
            messages_text.append(f"({msg_num})", style=THEME["text_dim"])
            messages_text.append(
                f" → {messages_dict[msg_num]}\n", style=THEME["text_primary"]
            )
            messages_text.append("\n")

        console.print(messages_text)
        console.print("─" * 80)

    # ==================== OVERALL STATUS ====================
    overall = diagnosis.get("overall", {})
    overall_status = overall.get("status", "❓")
    overall_message = overall.get("message", "No information")

    console.print("\n" + "─" * 80)
    overall_panel = Panel(
        Text(
            f"{overall_status} {overall_message}",
            justify="center",
            style=THEME["subtitle_color"],
        ),
        title="[bold]OVERALL[/bold]",
        border_style=(
            THEME["error"]
            if overall_status == "🔴"
            else THEME["warnings"] if "⚠️" in overall_status else THEME["ok"]
        ),
        padding=(1, 2),
    )
    console.print(overall_panel)
    console.print("─" * 80)

    console.print("\n")


def display_diagnosis_simple(diagnosis: dict, verbose: int = 3):
    """Display simple sin Rich - Adaptado a estructura real"""

    verbose_priorities = {
        1: ["❓", "🔴"],
        2: ["🔴", "🟡", "⚠️", "⚠️⚠️"],
        3: None,
        4: None,
    }
    priorities = verbose_priorities.get(verbose, None)
    messages = []
    msg_counter = 0

    def should_show(status):
        return priorities is None or status in priorities

    def safe_extract(data):
        """Extrae label/status/message"""
        if isinstance(data, dict):
            label = data.get("label", "N/A")
            status = data.get("status", "❓")
            message = data.get("message", "")
            return label, status, message
        return str(data), "❓", ""

    # TÍTULO
    print("\n\n" + "=" * 80)
    print("🧠  SANITY MONITOR".center(80))
    print("=" * 80)

    # OVERALL
    o_status = diagnosis["overall"]["status"]
    o_msg = diagnosis["overall"]["message"]
    print(f"\n{o_status} OVERALL: {o_msg}")
    print("-" * 80)

    # GRADIENTS
    if "gradients" in diagnosis:
        print("\n🧮 GRADIENTS")
        grad = diagnosis["gradients"]

        # Global norm
        g_label, g_status, g_msg = safe_extract(grad["global_norm"])
        if should_show(g_status):
            msg_num = msg_counter + 1 if verbose == 4 else None
            print(
                f"  {g_status} Global: {g_label}" + (f" ({msg_num})" if msg_num else "")
            )
            if verbose == 4 and g_msg:
                messages.append((msg_num, g_msg))
                msg_counter += 1

        # Norm per sample (mean + std)
        print("  📈 Per-sample:")
        nmean_label, nmean_status, nmean_msg = safe_extract(
            grad["norm_per_sample"]["mean"]
        )
        if should_show(nmean_status):
            msg_num = msg_counter + 1 if verbose == 4 else None
            print(
                f"    {nmean_status} Mean: {nmean_label}"
                + (f" ({msg_num})" if msg_num else "")
            )
            if verbose == 4 and nmean_msg:
                messages.append((msg_num, nmean_msg))
                msg_counter += 1

        nstd_label, nstd_status, nstd_msg = safe_extract(grad["norm_per_sample"]["std"])
        if should_show(nstd_status):
            msg_num = msg_counter + 1 if verbose == 4 else None
            print(
                f"    {nstd_status} Std:  {nstd_label}"
                + (f" ({msg_num})" if msg_num else "")
            )
            if verbose == 4 and nstd_msg:
                messages.append((msg_num, nstd_msg))
                msg_counter += 1

        # ReIm norms
        print("  🔄 Re/Im:")
        for net_name in ["global", "ModulusNet", "PhaseNet"]:
            r_label, r_status, r_msg = safe_extract(grad["ReIm_norms"][net_name])
            if should_show(r_status):
                msg_num = msg_counter + 1 if verbose == 4 else None
                print(
                    f"    {r_status} {net_name}: {r_label}"
                    + (f" ({msg_num})" if msg_num else "")
                )
                if verbose == 4 and r_msg:
                    messages.append((msg_num, r_msg))
                    msg_counter += 1

        # Percentage intervals (label es lista)
        perc_data = grad["percentage_norm_intervals"]
        perc_status = perc_data["status"]
        perc_labels = perc_data["label"]  # Lista de strings
        perc_msg = perc_data["message"]

        print(f"\n  📊 Gradient Histogram {perc_status}")
        for line in perc_labels:
            print(f"    {line}")

        if verbose == 4 and perc_msg:
            msg_num = msg_counter + 1
            messages.append((msg_num, perc_msg))
            msg_counter += 1

    # SAMPLING
    if "sampling" in diagnosis:
        print("\n📈 SAMPLING")
        for subkey, subdata in diagnosis["sampling"].items():
            s_label, s_status, s_msg = safe_extract(subdata)
            if should_show(s_status):
                msg_num = msg_counter + 1 if verbose == 4 else None
                print(
                    f"  {s_status} {subkey}: {s_label}"
                    + (f" ({msg_num})" if msg_num else "")
                )
                if verbose == 4 and s_msg:
                    messages.append((msg_num, s_msg))
                    msg_counter += 1

    # SAMPLING HISTOGRAM
    if "sample_histogram" in diagnosis:
        print("\n📊 SAMPLE HISTOGRAM")
        hist_data = diagnosis["sample_histogram"]["histogram"]["label"]
        magnetization = hist_data["magnetization"]
        counts = hist_data["counts"]
        for m, count in zip(magnetization, counts):
            print(f"{m} --> {count}")

    # PHASE
    if "phase" in diagnosis:
        print("\n⚛️  PHASE")
        for subkey, subdata in diagnosis["phase"].items():
            p_label, p_status, p_msg = safe_extract(subdata)
            if should_show(p_status):
                msg_num = msg_counter + 1 if verbose == 4 else None
                print(
                    f"  {p_status} {subkey}: {p_label}"
                    + (f" ({msg_num})" if msg_num else "")
                )
                if verbose == 4 and p_msg:
                    messages.append((msg_num, p_msg))
                    msg_counter += 1

    # MESSAGES verbose=4
    if verbose == 4 and messages:
        print("\n" + "─" * 80)
        print("📝 MESSAGES".center(80))
        print("─" * 80)
        for num, msg in sorted(messages):
            print(f"({num:2d}) {msg}")

    print("\n")


if __name__ == "__main__":
    # Example usage
    diagnosis_example = {
        "gradients": {
            "norm_per_sample": {
                "mean": {
                    "label": "⟨||∇logψ||⟩_s = 7.83e-03",
                    "status": "✅",
                    "message": "SANO. Gradientes normales → Entrenando bien",
                },
                "std": {
                    "label": "σ[||∇logψ||] = 1.41e+01  (σ/μ=1796.10)",
                    "status": "🔴",
                    "message": "Muy alta variabilidad, sampling problemático!",
                },
            },
            "global_norm": {
                "label": "||∇logψ||_global = 6.39e+01",
                "status": "⚠️",
                "message": "Gradientes grandes, posible inestabilidad.",
            },
            "ReIm_norms": {
                "global": {
                    "label": "||Re(∇ψ)|| / ||Im(∇ψ)|| = 1.97e+03 / 2.11e+03 = 0.93",
                    "status": "🟢",
                    "message": "Razón Re/Im equilibrada en el global.",
                },
                "ModulusNet": {
                    "label": "||∇ψ_ph||_Re / ||∇ψ_ph||_Im = 5.73e+03 / 4.85e-02 = 118166.10",
                    "status": "🟢",
                    "message": "Gradientes del módulo dominantes y estables.",
                },
                "PhaseNet": {
                    "label": "||∇ψ_ph||_Im / ||∇ψ_ph||_Re = 5.23e+03 / 3.21e-02 = 162884.72",
                    "status": "🟢",
                    "message": "Gradientes de la fase dominantes y estables.",
                },
            },
            "percentage_norm_intervals": {
                "label": [
                    "   > 1e+03:        0.000%",
                    "1e+02 <--> 1e+03:   0.000%",
                    "1e+01 <--> 1e+02:   0.000%",
                    "1e+00 <--> 1e+01:   8.501%",
                    "1e-01 <--> 1e+00:   42.733%",
                    "1e-02 <--> 1e-01:   40.754%",
                    "1e-03 <--> 1e-02:   6.953%",
                    "1e-04 <--> 1e-03:   0.955%",
                    "    < 1e-04:        0.104%",
                ],
                "status": "🟢",
                "message": "Distribución de normas dentro de rangos aceptables",
            },
        },
        "sampling": {
            "acceptance": {
                "label": "Acc = 0.486",
                "status": "🟢",
                "message": "Acceptance OK",
            },
            "tau_corr": {
                "label": "τ_corr = 0.19",
                "status": "🟢",
                "message": "Autocorrelación baja",
            },
            "ESS": {
                "label": "ESS / N (eff)= 1488 / 2048 (0.73)",
                "status": "🟢",
                "message": "Eficiencia aceptable",
            },
        },
        "phase": {
            "mean": {
                "label": "|⟨e^(i·ϕ)⟩| = 0.025",
                "status": "🔴",
                "message": "Phase problem GRAVE → Revisa ansatz",
            },
            "std": {
                "label": "σ[e^(i·ϕ)] = 1.000",
                "status": "🟢",
                "message": "Varianza de fase aceptable",
            },
        },
        "overall": {"status": "🔴", "message": "Intervenir"},
    }

    verbose = 0

    if verbose != 0:
        display_diagnosis_sanity_monitor(diagnosis_example, verbose)
    else:
        display_diagnosis_simple(diagnosis_example, verbose)
