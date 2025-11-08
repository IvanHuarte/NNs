import numpy as np
import jax
import jax.numpy as jnp
import netket as nk


def generate_training(setup):

    lr_name = setup['lr_name']
    training_setup = setup['setup']
    lr_setup = setup['lr_schedules'][lr_name]
    

    if lr_name == "segment":
        
        segments = []
        modes = []
        segments_raw, modes_raw, repeats_raw = (
            training_setup["segments"],
            training_setup["mode"],
            training_setup["repeat_segment"],
        )
        lr_raw = lr_setup["lr"]


        for seg, mode, repeats in zip(segments_raw, modes_raw, repeats_raw):
            segments += [seg] * repeats
            modes += [mode] * repeats

        lr_segments = [[lr_raw[i]]*len(segment_modes) for i, segment_modes in enumerate(modes)]

        assert len(segments_raw) == len(modes_raw) == len(lr_segments),(
                 f"Las listas de segmentos, modos y repeticiones deben tener la misma longitud"
                 f",pero tienen longitudes {len(segments_raw)}, {len(modes_raw)} y {len(lr_segments)}"
        )

        return segments, modes, lr_segments


def get_ST_schedule(name, setup):
    """
    Returns a learning rate schedule as an array whose elements are a constant
    learning rate in each segment.

    """

    if name == "segment":
        return jnp.array(setup["lr"])

    elif name == "stairs_schedule":
        lr_schedule = jnp.logspace(
            start=jnp.log10(setup["lr0"]),
            stop=jnp.log10(setup["lr_min"]),
            num=setup["total_segments"],
        )
        return lr_schedule
