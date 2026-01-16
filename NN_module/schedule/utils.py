

def get_schedule_label(split_training, setup):

    if split_training:
        label = "ST"

        epo_st = setup["epochs_struct"]
        mod_st = setup["modes_struct"]
        epo_st = [period for eon in epo_st for era in eon for period in era]
        mod_st = [period for eon in mod_st for era in eon for period in era]

        for epoch, modes in zip(epo_st, mod_st):
            label += "_"
            label += f"{epoch}{modes}"

    else:
        label = "Both"

    return label


def schedule_from_array(x):
    def schedule_fun(step):
        return x[step]

    return schedule_fun

