import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, AnyStr, Callable

from NN_module.NN_utils import traslations_2D


def final_ensemble(ensem_mode: AnyStr = "sum") -> Callable:

    # Operation selection
    if ensem_mode == "sum":
        return jnp.sum
    elif ensem_mode == "mean":
        return jnp.mean


class Transversal_Worker(nn.Module):
    """
    Flax module to train module and phase separately
    """

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B = x.shape[0]
        # print(f"x_in: {x.shape}")

        x = jnp.stack([module(x) for module in self.Trans], axis=0)
        # print(f"res: {x.shape}")

        # Norm
        if self.post_norm:
            x = x.swapaxes(0, -1)
            x = nn.LayerNorm()(x)
            x = x.swapaxes(0, -1)

        else:
            self.norm = lambda x: x

        # print(f"x_norm: {x.shape}")

        x = final_ensemble(ensem_mode=self.operation)(x, axis=0, keepdims=True)
        # print(f"final_ensemble: {x.shape}")
        x = jnp.atleast_2d(x).reshape(B, *x.shape[2:])

        # print(f"x_group: {x.shape}")
        # print(f"\n")
        return self.squeeze(x)


class Transversal_2D(nn.Module):

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False
    squeeze: Callable = lambda x: x

    lattice_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x):

        worker = Transversal_Worker(
            Trans=self.Trans,
            operation=self.operation,
            post_norm=self.post_norm,
            squeeze=self.squeeze,
        )

        # 2D traslation
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=None, memory=False
        )

        return jax.vmap(worker, in_axes=0)(traslational_x).mean(axis=0)


class Transversal_Z2(nn.Module):

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False
    squeeze: Callable = lambda x: x

    "Symmetries"
    symm_2D: bool = False
    trivial_Z2: bool = False

    lattice_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            worker = Transversal_2D(
                Trans=self.Trans,
                operation=self.operation,
                post_norm=self.post_norm,
                lattice_size=self.lattice_size,
                squeeze=self.squeeze,
            )
        else:
            worker = Transversal_Worker(
                Trans=self.Trans,
                operation=self.operation,
                post_norm=self.post_norm,
                squeeze=self.squeeze,
            )

        output_x = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        # Concatenamos las dos contribuciones
        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=0)
            return res
        else:
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)
            return res


class Transversal(nn.Module):
    """
    Flax module to have a general model (Trans) which carries and trains
    global information of the system and ending up with an ending model
    which carries the dimensional reduction and/or modulus-phase spliting.
    """

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False
    squeeze: Callable = lambda x: x

    "Symmetries"
    symm_Z2: bool = False
    symm_2D: bool = False
    trivial_Z2: bool = True

    "Needed for performing 2D traslation symmetries"
    lattice_size: Tuple[int, int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = Transversal_Z2(
                Trans=self.Trans,
                operation=self.operation,
                post_norm=self.post_norm,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                lattice_size=self.lattice_size,
                squeeze=self.squeeze,
            )
        elif self.symm_2D:
            worker = Transversal_2D(
                Trans=self.Trans,
                operation=self.operation,
                post_norm=self.post_norm,
                lattice_size=self.lattice_size,
                squeeze=self.squeeze,
            )
        else:
            worker = Transversal_Worker(
                Trans=self.Trans,
                operation=self.operation,
                post_norm=self.post_norm,
                squeeze=self.squeeze,
            )

        x = worker(x)

        return x
