import sys
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import netket as nk
import numpy.typing as npt
import optax

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


def circulant(row: npt.ArrayLike, times: Optional[int] = None) -> npt.ArrayLike:
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
    x: npt.ArrayLike, stride: Tuple[int, int] = (1, 1)
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

    # idx = jnp.unravel_index(jnp.arange(H * W), (H, W))
    # shift_idx = jnp.array(idx).T
    # mask = (shift_idx[:, 0] % stride[0] == 0) & (shift_idx[:, 1] % stride[1] == 0)
    # jax.debug.print("Mask: {}", mask)
    # shift_idx = shift_idx[mask].reshape(Ht, Wt, 2)
    # jax.debug.print("Idx: {}", shift_idx)

    roll = lambda x, shift: jnp.roll(x, shift=shift, axis=(-3, -2))

    return jax.vmap(jax.vmap(roll, in_axes=(None, 0)), in_axes=(None, 0))(x, shift_idx)


def Translations2D_scan(x: npt.ArrayLike, stride: Tuple[int, int]) -> npt.ArrayLike:
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
    stride: Tuple[int, int] = (1, 1),
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
