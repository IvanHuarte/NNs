#!/usr/bin/env python

import itertools

import numpy as onp
import numpy.typing as npt
import jax
import jax.typing as jt
import jax.random
import jax.numpy as jnp
import flax
import flax.linen as nn


DTYPE = jnp.float32


def make_circulant(tile: npt.ArrayLike, roll_axis: int = -1) -> npt.ArrayLike:
    """Add an axis to an array making it circulant.

    Args:
        tile: the building block of the circulant array.
        roll_axis: axis over which to roll the tile each time it is repeated.

    Returns:
        A new array with all the offset versions of the tile.
    """
    tile = jnp.asarray(tile)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1, axis=roll_axis)
        return (new_carry, new_carry)

    nruter = jax.lax.scan(scan_arg, tile, length=tile.shape[roll_axis])[1][
        ::-1, ...
    ]

    return nruter


def create_2D_circulant_from_motif(motif: npt.ArrayLike) -> npt.ArrayLike:
    """Create a circulant attention array for a 2D system from a 2D motif.

    Args:
        motif: the repeating motif.

    Returns:
        The attention (a 4D array).
    """
    with_three_axes = make_circulant(motif, 1)
    with_four_axes = make_circulant(with_three_axes, 1)

    return with_four_axes


class AffinityPosWeight2D(nn.Module):
    "Flax module that implements a circular positional attention in 2D."

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        weight_motif = self.param(
            "alpha_delta",
            nn.initializers.glorot_normal(),
            (x.shape[-3], x.shape[-2]),
            DTYPE,
        )
        weight = create_2D_circulant_from_motif(weight_motif)
        return jnp.einsum("ijkl,klm->ijm", weight, x)


class PositionalHead2D(nn.Module):
    """Flax module that implements a single head of linearized attention in 2D.

    Args:
        head_size: The dimension of the head.
    """

    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        value = nn.Dense(self.head_size, use_bias=False, param_dtype=DTYPE)
        aff = AffinityPosWeight2D()

        return aff(value(x))


if __name__ == "__main__":
    HEAD_SIZE = 16
    fake_input = jnp.asarray([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])
    fake_input = fake_input.reshape(
        (fake_input.shape[0], fake_input.shape[1], 1)
    )
    rng = jax.random.key(31337)

    model = PositionalHead2D(HEAD_SIZE)
    sacrificial_rng, rng = jax.random.split(rng)
    params = model.init(sacrificial_rng, fake_input)

    original_output = model.apply(params, fake_input)
    for delta_i, delta_j in itertools.product(
        range(fake_input.shape[0]), range(fake_input.shape[1])
    ):
        displaced_input = jnp.roll(fake_input, (delta_i, delta_j), axis=(0, 1))
        new_output = model.apply(params, displaced_input)
        displaced_output = jnp.roll(
            original_output, (delta_i, delta_j), axis=(0, 1)
        )
        if not jnp.allclose(new_output, displaced_output):
            sys.exit("Not equivariant!")

    print("Equivariant")
