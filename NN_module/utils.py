import os
import sys
import json

from pathlib import Path
from copy import deepcopy

import numpy.typing as npt
import jax
import jax.numpy as jnp
import optax
import netket as nk


def print_tree(tree, prefix="", values=False):
    for key, val in tree.items():

        if isinstance(val, dict):
            print(prefix + str(key))
            print_tree(val, prefix + "   ", values=values)
        else:
            if values:
                print(prefix + f"{key!s}: {val!s}")
            else:
                print(prefix + str(key))


def compare_params(old_params, new_params, atol=1e-13):
    def compare_fn(p_old, p_new):
        return not jax.numpy.allclose(p_old, p_new, atol=atol)

    diffs = jax.tree_util.tree_map(compare_fn, old_params, new_params)

    print("Ha cambiado: (True) //  No ha cambiado: (False) \n\n")
    print_tree(diffs, values=True)
    print("\n")


sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent
        / "Transformers/transformer_LR_WF_public"
    )
)

from transformer_LR_WF.utils import InvertMagnetization

REAL_DTYPE = jnp.asarray(1.0).dtype

optimizer_name_dict = {
    "Sgd": nk.optimizer.Sgd,
    "adam": nk.optimizer.Adam,
    "AdaGrad": nk.optimizer.AdaGrad,
}

sampler_dict = {
    "MetropolisLocal": nk.sampler.MetropolisLocal,
    "MetropolisExchange": nk.sampler.MetropolisExchange,
    "MetropolisHamiltonian": nk.sampler.MetropolisHamiltonian,
    "MetropolisSampler": nk.sampler.MetropolisSampler,
}

rule_dict = {
    "LocalRule/InvertMagnetization": nk.sampler.rules.MultipleRules(
        [nk.sampler.rules.LocalRule(), InvertMagnetization()], [0.75, 0.25]
    ),
}


def cos_exp_scheduler(epochs, lr0, decay, cycles, n, lr_min):

    exp = lambda step, a: jnp.exp(-a * step / epochs)
    cos = lambda step, cycles: jnp.cos(step * 2 * jnp.pi / epochs * (cycles - 0.5))
    line = lambda step, n: (lr_min - n) / epochs * step + n

    def scheduler_callable(step):
        return (lr0 - n) * exp(step, decay) * (1 + cos(step, cycles)) / 2 + line(
            step, n
        )

    return scheduler_callable


def scheduler_initializer(name, setup):

    print(setup["total_epochs"])
    epochs = setup["total_epochs"]

    if name == "cos_exp_scheduler":
        return cos_exp_scheduler(
            epochs=epochs,
            lr0=setup["lr0"],
            decay=setup["decay_exp"],
            cycles=setup["cosine_cycles"],
            n=setup["n"],
            lr_min=setup["lr_min"],
        )
    elif name == "warmup_exponential_decay":
        return optax.warmup_exponential_decay_schedule(
            init_value=setup["lr0"],
            peak_value=setup["peak_value"],
            warmup_steps=setup["warmup_steps"],
            transition_steps=1,
            decay_rate=setup["decay_rate"],
        )


####################################################################################
#                                                                                  #
#                                   SIMETRIES                                      #
#                                                                                  #
####################################################################################


def circulant(row: npt.ArrayLike, times: int | None = None) -> npt.ArrayLike:
    """Build a (full or partial) circulant matrix based on an array.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        If `times` is None, a square matrix with all the offset versions of the
        first argument. Otherwise, `times` rows of a circulant matrix.
    """
    row = jnp.asarray(row)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1)
        return (new_carry, new_carry)

    if times is None:
        nruter = jax.lax.scan(scan_arg, row, row)[1][::-1, :]
    else:
        nruter = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return nruter


def Translations2D_vmap(
    x: npt.ArrayLike, stride: tuple[int, int] = (1, 1)
) -> npt.ArrayLike:
    """
    Takes a 4-tensor of shape (B, H, W, Ch) and returns a tensor of shape
    (Ht, Wt, B, H, W, Ch) which represtents the traslation replicas of the input.
    Enhances time execution.
    """
    _, H, W, _ = x.shape
    # Ht, Wt = H // stride[0], W // stride[1]

    ys = jnp.arange(0, H, stride[0])
    xs = jnp.arange(0, W, stride[1])

    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    shift_idx = jnp.stack([yy, xx], axis=-1)

    roll = lambda x, shift: jnp.roll(x, shift=shift, axis=(-3, -2))

    return jax.vmap(jax.vmap(roll, in_axes=(None, 0)), in_axes=(None, 0))(x, shift_idx)


def Translations2D_scan(x: npt.ArrayLike, stride: tuple[int, int]) -> npt.ArrayLike:
    """

    Takes a 4-tensor of shape (B, H, W, Ch) and returns a tensor of shape
    (Ht, Wt, B, H, W, Ch) which represtents the traslation replicas of the input.
    Enhances memory saving.
    """
    _, H, W, _ = x.shape
    Ht, Wt = H // stride[0], W // stride[1]

    def scan_and_roll_x(carry_x, _):

        def scan_and_roll_y(carry_y, _):
            y = jnp.roll(carry_y, shift=-stride[1], axis=-2)
            return y, y

        _, block_y = jax.lax.scan(scan_and_roll_y, carry_x, length=Wt)
        x = jnp.roll(carry_x, shift=stride[0], axis=-3)

        return x, block_y[::-1]

    return jax.lax.scan(scan_and_roll_x, x, length=Ht)[1]


def Translations2D(
    x: npt.ArrayLike,
    stride: tuple[int, int] = (1, 1),
    save_memory: bool = False,
) -> npt.ArrayLike:
    """
    Returns 2D translations of an input x=(B, H, W, Ch) and returns the translations set
    as x = (B, Ht, Wt, H, W, Ch), where Ht and Wt are the considered translations of the
    lattice depending on the value of stride.
    """

    x = jnp.atleast_2d(x)

    if save_memory:
        x = Translations2D_scan(x, stride)
    else:
        x = Translations2D_vmap(x, stride)

    x = x.transpose(
        (2, 0, 1, 3, 4, 5)
    )  # (Ht, Wt, B, H, W, Ch) ---> (B, Ht, Wt, H, W, Ch)

    return x


####################################################################################
#                                                                                  #
#                               CUSTOM FUNCTIONS                                   #
#                                                                                  #
####################################################################################

from jax._src.typing import Array, ArrayLike

eps = 1e-8


@jax.custom_jvp
def _atan2(y, x):
    "Wrapper for the regular atan2 function"
    return jax.lax.atan2(y, x)


@_atan2.defjvp
def _atan2_jvp(primals, tangents):
    "Custom jvp for _atan2 that avoids the singularity at z = 0 + 0j"

    y, x = primals
    ydot, xdot = tangents

    primal_out = _atan2(y, x)
    z_2 = x * x + y * y

    tangent_out = jnp.where(z_2 == 0, 0.0, (-y * xdot + x * ydot) / z_2)

    return (primal_out, tangent_out) * jnp.angle


@jax.custom_jvp
def _angle(z: ArrayLike) -> Array:
    return jnp.angle(z)


@_angle.defjvp
def _angle_jvp(primals, tangents):
    (z,) = primals
    (zdot,) = tangents

    out = jnp.angle(z)

    x, y = z.real, z.imag

    dx, dy = zdot.real, zdot.imag

    z2 = x * x + y * y

    dout = jnp.where(
        z2 < eps,
        0.0,
        (-y * dx + x * dy) / jnp.where(z2 < eps, 1.0, z2),
    )

    return out, dout


####################################################################################
#                                                                                  #
#                               SIMULATION STUFF                                   #
#                                                                                  #
####################################################################################


def save_config_files(uuid_folder, configuration_dicts, configuration_names, size):
    configuration_basenames = [Path(file_path).name for file_path in configuration_names]
    sizes = [size]
    write_folder = uuid_folder + "config_files/"

    os.makedirs(write_folder, exist_ok=True)

    for config_name, config_dict in zip(configuration_basenames, configuration_dicts):
        config_tmp = deepcopy(config_dict)

        if "sizes" in config_tmp:
            config_tmp["sizes"] = sizes

        with open(write_folder + config_name, "w") as f:
                json.dump(config_tmp, f, separators=(",", ":"), sort_keys=True, indent=4)
        
    