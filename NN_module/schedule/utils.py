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

def all_submodules(submodules, lr):

    if len(submodules) == len(lr):
        return lr

    if len(lr) == 1:
        print(submodules)
        return lr * len(submodules)
    
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
        mode = submodules
        lr = all_submodules(submodules, lr)

    else:
        assert len(mode) == len(lr)
        mode, lr = some_submodules(submodules, mode, lr)

    return mode, lr
    

def schedule_from_array(x):
    def schedule_fun(step):
        return x[step]

    return schedule_fun

def arch_recognizer(params):
    labels = list(params.keys())

    if all(key in ['ModulusNet', 'PhaseNet'] for key in labels):
        return "SplitTraining"
    
    elif all('Seq' in key or 'End' in key for key in labels):
        return "Sequential"
    
    elif all('Trans' in key for key in labels):
        return "Transversal"
    
    else:
        raise NotImplementedError(f"There is no mask for this architecture: {labels}")
    
