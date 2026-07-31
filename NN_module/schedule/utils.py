import jax
import jax.numpy as jnp
import flax
import ast

from NN_module.schedule import SCHEDULES
from NN_module.utils import print_tree


def get_schedule_label(setup):

    epo_st = setup["epochs_struct"]
    mod_st = setup["modes_struct"]
    epo_st = [period for eon in epo_st for era in eon for period in era]
    mod_st = [period for eon in mod_st for era in eon for period in era]

    label = "ST"
    for epoch, modes in zip(epo_st, mod_st):
        label += "_"
        label += f"{epoch}{modes}"

    for char in ["[", "]", "'", '"']:
        label = label.replace(char, "") if char != "[" else label.replace("[", "-")

    # Filter characters
    label = label.replace(",", "-")
    label = label.replace(" ", "")

    return label


##########################################
# FUNCTIONS FOR DECODE MODES INTO PATHS
# AND MATCH LR-MODES LENGHT
##########################################


def all_submodules(submodules, lr):

    if len(lr) == 1:
        return lr * len(submodules)

    if len(submodules) == len(lr):
        return lr


def some_submodules(submodules, mode, lr):
    mode = [submodules[idx] for idx in mode]
    if len(lr) == 1:
        lr = lr * len(mode)
    else:
        assert len(lr) == len(mode)

    return mode, lr


def decode_arch_labels(submodules, mode, lr):
    assert isinstance(mode, (list, tuple))

    if mode[0] == "A":
        assert len(mode) == 1
        mode = [mod for k, mod in submodules.items() if len(k) == 1]
        lr = all_submodules(mode, lr)

    else:
        assert len(mode) == len(lr)
        mode, lr = some_submodules(submodules, mode, lr)

    return mode, lr


def period_from_string(epochs, lr_instruction):
    schedule_name, str_args = lr_instruction.split("(")
    schedule = SCHEDULES[schedule_name]
    args = ast.literal_eval("(" + str_args)

    period = schedule(epochs, *args)
    return period


def generate_period(epochs, mode, lr_instruction):

    if isinstance(lr_instruction, (list, tuple)):
        period = []
        info = []
        for lr_ins, mod in zip(lr_instruction, mode):
            nruter = generate_period(epochs, mod, lr_ins)
            period.append(nruter[0])
            info.append(nruter[1])

    elif isinstance(lr_instruction, (int, float)):
        period = jnp.array([lr_instruction] * epochs)
        info = (epochs, mode, lr_instruction)

    elif isinstance(lr_instruction, str):
        period = period_from_string(epochs, lr_instruction)
        info = (epochs, mode, lr_instruction)

    else:
        raise TypeError(f"Unsupported lr_instruction: {lr_instruction}")

    return period, info


#################################################
# WRAPPER WHICH CONVERTS AN ARRAY INTO CALLABLE
#################################################


def schedule_from_array(x):
    def schedule_fun(step):
        return x[step]

    return schedule_fun


#################################################
# CHANGE NN ARCHITECTURE CONTROL
#################################################


def eon_change(i_eon, i_era, i_per, n_eons, n_eras, n_pers):

    eon_bool = i_eon != n_eons - 1
    era_bool = i_era == n_eras - 1
    per_bool = i_per == n_pers - 1

    return eon_bool & era_bool & per_bool


################################################
# CALCULATE LEARNING RATE SCHEDULE FROM SETUP
################################################


def calculate_lr_schedule(setup):

    epo_st = setup["epochs_struct"]
    mod_st = setup["modes_struct"]
    lr_st = setup["lr_struct"]

    epo_st = [period for eon in epo_st for era in eon for period in era]
    mod_st = [period for eon in mod_st for era in eon for period in era]
    lr_st = [period for eon in lr_st for era in eon for period in era]

    lr_schedule = []
    info_schedule = []
    for epochs, mode, lr_instruction in zip(epo_st, mod_st, lr_st):
        period, info = generate_period(epochs, mode, lr_instruction)
        lr_schedule.append(*period)
        info_schedule.append(*info)

    return lr_schedule, info_schedule
