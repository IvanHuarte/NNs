"""Frozen copy of the pre-refactor labels (reference for test_labels)."""

from datetime import datetime


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
    elif cm_name == "SSMSquare":
        J = cm_setup["params"]["J"]
        Jp = cm_setup["params"]["Jp"]
        fields = cm_setup["params"]["fields"]
        cparams = f"_JJp_{J}_{Jp}_XYZ_{fields[0]}_{fields[1]}_{fields[2]}"
        call_params = r"$J=%.2f$  $J_p=%.2f$  $XYZ=(%.1f,%1.f,%.1f)$" % (
            J,
            Jp,
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
        "Callback  " + model_label + "  " + call_params + f"  ({size[0]}x{size[1]})"
    )

    return sim_label, ED_label, json_label, title_label_callback
