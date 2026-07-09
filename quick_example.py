#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import itertools
import sys

import jax
import jax.numpy as jnp
import numpy.typing as npt


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

    nruter = jax.lax.scan(scan_arg, tile, length=tile.shape[roll_axis])[1][::-1, ...]

    return nruter


if __name__ == "__main__":
    # Create an array with four axes and the circulant property we are after.
    motif = jnp.asarray([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
    with_three_axes = make_circulant(motif, 1)
    print(f"3axes: {with_three_axes}")
    with_four_axes = make_circulant(with_three_axes, 1)
    print(f"4axes: {with_four_axes}")

    print(with_four_axes.reshape(3 * 3, 3 * 3))

    # Check that it have 2D translation symmetry.
    for i_origin, j_origin in itertools.product(
        range(motif.shape[0]), range(motif.shape[1])
    ):
        reference = with_four_axes[0, 0, i_origin, j_origin]
        for delta_i, delta_j in itertools.product(
            range(motif.shape[0]), range(motif.shape[1])
        ):
            element = with_four_axes[
                delta_i,
                delta_j,
                (i_origin + delta_i) % motif.shape[0],
                (j_origin + delta_j) % motif.shape[1],
            ]
            if reference != element:
                sys.exit("Error: not circulant.")

    print("Success!")
