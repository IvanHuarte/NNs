import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, AnyStr, Callable

from NN_module.NN_utils import traslations_2D
from NN_module.models.toolbox import CarreteSign, AddPhase

def final_ensemble(ensem_mode: AnyStr = "sum") -> Callable:

    # Operation selection
    if ensem_mode == "sum":
        return jnp.sum
    elif ensem_mode == "sum_angles":

        def fun(x, axis=0, keepdims=True):
            x = jnp.sum(x, axis=axis, keepdims=keepdims)
            x = (x + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return x

        return fun

    elif ensem_mode == "product":
        return jnp.prod

    elif ensem_mode == "mean":
        return jnp.mean
    elif ensem_mode == "modulus_marshall":

        def fun(x, axis=0, keepdims=True):
            x = jnp.log(x[1] * jnp.exp(x[0]))
            return x.astype(jnp.complex128)

        return fun

    else:
        raise ValueError(f"Ensemble mode {ensem_mode} not recognized.")


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

        x = jnp.atleast_2d(x)

        B = x.shape[0]
        # print(f"x_in: {x.shape}")

        x = jnp.stack([module(x) for module in self.Trans], axis=0)
        # print(f"res: {x.shape}")

        # Norm
        if self.post_norm:
            x = nn.LayerNorm()(x.swapaxes(0, -1)).swapaxes(0, -1)

        # print(f"x_norm: {x.shape}")

        x = final_ensemble(ensem_mode=self.operation)(x, axis=0, keepdims=True)
        # jax.debug.print("x_after: {} \n\n", x)

        # print(f"final_ensemble: {x}")
        x = jnp.atleast_2d(x).reshape(B, *x.shape[2:])

        # print(f"x_group: {x.shape}")
        # print(f"\n")
        return self.squeeze(x)


class Transversal_2D(nn.Module):

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        worker = Transversal_Worker(
            Trans=self.Trans,
            operation=self.operation,
            post_norm=self.post_norm,
            squeeze=self.squeeze,
        )

        na, nb = jnp.unravel_index(
            jnp.arange(self.lattice_size[0] * self.lattice_size[1]), self.lattice_size
        )

        characters = jnp.exp(
            2j
            * jnp.pi
            * (
                self.irrep[0] * na / self.lattice_size[0]
                + self.irrep[1] * nb / self.lattice_size[1]
            )
        )

        # 2D traslation
        traslational_x = traslations_2D(
            x, size=self.lattice_size, token_size=self.token_size, memory=False
        )
    
        ffw = jax.vmap(worker, in_axes=0)(traslational_x).T

        x = ffw * characters

        x = jnp.atleast_1d(
            x.mean(axis=-1)
        )

        return x


class Transversal_2DAnchor(nn.Module):

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False

    lattice_size: Tuple[int, int] = None
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        worker = Transversal_Worker(
            Trans=self.Trans,
            operation=self.operation,
            post_norm=self.post_norm,
            squeeze=self.squeeze,
        )

        # 2D traslational anchoring
        x, anchors = CarreteSign(lattice_size=self.lattice_size, irrep=self.irrep)(x)

        x = jnp.atleast_1d(worker(x))

        # Add phase according to the irrep and the anchor
        x = AddPhase(lattice_size=self.lattice_size, irrep=self.irrep)(x, anchors)

        return x


class Transversal_Z2(nn.Module):

    Trans: Tuple[nn.Module, ...]
    operation: str = "sum"
    post_norm: bool = False

    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    "Symmetries"
    symm_2D: bool = False
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    use_anchor: bool = False

    trivial_Z2: bool = False

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.symm_2D:
            if self.use_anchor:
                worker = Transversal_2DAnchor(
                    Trans=self.Trans,
                    operation=self.operation,
                    post_norm=self.post_norm,
                    lattice_size=self.lattice_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
            
            else:
                worker = Transversal_2D(
                    Trans=self.Trans,
                    operation=self.operation,
                    post_norm=self.post_norm,
                    lattice_size=self.lattice_size,
                    token_size=self.token_size,
                    irrep=self.irrep,
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

    "Symmetries"
    symm_2D: bool = False
    irrep: Tuple[int] = (0, 0)  # Tuple (q_1, q_2) representing the irrep.
    use_anchor: bool = False

    symm_Z2: bool = False
    trivial_Z2: bool = True

    "Needed for performing 2D traslation symmetries"
    lattice_size: Tuple[int, int] = None
    token_size: Tuple[int, int] = None

    squeeze: Callable = lambda x: x

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.symm_Z2:
            worker = Transversal_Z2(
                lattice_size=self.lattice_size,
                Trans=self.Trans,
                operation=self.operation,
                post_norm=self.post_norm,
                trivial_Z2=self.trivial_Z2,
                symm_2D=self.symm_2D,
                irrep=self.irrep,
                use_anchor=self.use_anchor,
                token_size=self.token_size,
                squeeze=self.squeeze,
            )
        if self.symm_2D:
            if self.use_anchor:
                worker = Transversal_2DAnchor(
                    Trans=self.Trans,
                    operation=self.operation,
                    post_norm=self.post_norm,
                    lattice_size=self.lattice_size,
                    irrep=self.irrep,
                    squeeze=self.squeeze,
                )
            
            else:
                worker = Transversal_2D(
                    Trans=self.Trans,
                    operation=self.operation,
                    post_norm=self.post_norm,
                    lattice_size=self.lattice_size,
                    token_size=self.token_size,
                    irrep=self.irrep,
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
