"""
Floating point precision of the networks (config0 "network_dtype").

All the modules were written with float64 parameters and activations
(jax_enable_x64 is on). On consumer GPUs float64 runs at 1/32-1/64 of the
float32 rate, so the forward passes (local energy, sampling) dominate the
step time. set_network_dtype("float32") switches the dtype used by the
modules of the transformer networks (ViT2D, CvT3, toolbox, OutputHead /
DeepOutputHead) before the model is built. The momentum projection
(Sequivariant) and everything outside the network (SR, energies) stay in
float64/complex128.

Mixed precision: the parameters stay float64 (state_kwargs gives MCState an
init_fun that stores them in float64 and an apply_fun that casts them to
float32 on entry). Forward passes run in float32, while the Jacobian (taken
w.r.t. the float64 parameters), the NTK/Cholesky solve of minSR and the
optimizer work in float64. A pure float32 test on 6x6 (Nov 2026) was 7x
faster but blew up once diag_shift went below ~1e-3: the float32 SR solve,
not the network, was the problem.

Call it once, before building the network. Default "float64": nothing changes.

Matmul precision (config0 "matmul_precision", only float32 networks)
-----------------------------------------------------------------------
On Ampere and newer GPUs XLA computes float32 matmuls/convolutions with
TF32 tensor cores by default (10-bit mantissa). That is NOT accurate enough
for NQS: on the RTX 3080 (6x6, random init) TF32 gives errors in log psi up
to 0.17 (CvT3) / 0.12 (ViT2D) and a 3 % (CvT3) error in the Jacobian, and a
CvT3 run stalled at -46 instead of -72 (Nov 2026). True float32 ("highest")
gives 1e-4 in log psi and 2e-5 in the Jacobian. Hence the default here is
"highest"; float64 matmuls are not affected by this setting.
    "highest"        true float32 (default)
    "default"        XLA default = TF32 on tensor cores (fast, inaccurate)
    "bfloat16"       bf16 inputs, float32 accumulation (even less accurate)
"""

import importlib

import jax
import jax.numpy as jnp

DTYPES = {
    "float64": (jnp.float64, jnp.complex128),
    "float32": (jnp.float32, jnp.complex64),
}

# module path -> names of its dtype globals
_MODULES = {
    "Hydra.NN.toolbox": ("REAL_DTYPE",),
    "Hydra.NN.NN.SingleModules.ViT2D": ("DTYPE",),
    "Hydra.NN.NN.SingleModules.CvT3": ("DTYPE",),
    "Hydra.NN.NN.SingleModules.FluxFunc": ("DTYPE", "CDTYPE"),
}
_CLASSES = {  # classes with a (non-field) param_dtype class attribute
    "Hydra.NN.NN.SingleModules.FluxFunc": ("OutputHead", "DeepOutputHead"),
}

_current = "float64"


def set_network_dtype(name="float64"):
    """Set the dtype of the network modules ("float64" or "float32")."""
    global _current
    if name not in DTYPES:
        raise ValueError(f"network_dtype={name!r}, expected one of {list(DTYPES)}")
    real, cplx = DTYPES[name]
    for path, names in _MODULES.items():
        mod = importlib.import_module(path)
        for n in names:
            setattr(mod, n, cplx if n.startswith("C") else real)
    for path, classes in _CLASSES.items():
        mod = importlib.import_module(path)
        for c in classes:
            setattr(getattr(mod, c), "param_dtype", real)
    _current = name
    if name != "float64":
        print(f"[precision] network modules in {name}")
    return name


MATMUL_PRECISIONS = ("default", "tensorfloat32", "highest", "float32", "bfloat16")


def set_matmul_precision(name="highest"):
    """Default precision of float32 matmuls/convolutions (see module doc)."""
    if name not in MATMUL_PRECISIONS:
        raise ValueError(f"matmul_precision={name!r}, expected one of {MATMUL_PRECISIONS}")
    jax.config.update("jax_default_matmul_precision", None if name == "default" else name)
    if name != "highest":
        print(f"[precision] float32 matmuls: {name} (TF32/bf16: inaccurate for NQS)")
    return name


def get_network_dtype():
    return _current


def _cast(tree, dtype):
    return jax.tree_util.tree_map(
        lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, tree
    )


def _init64(model, rng, x, **kw):
    """
    Initialise exactly as the float64 network (same random draws for the
    same seed): the module dtype globals are read while tracing, so they
    are switched to float64 for the init and restored afterwards.
    """
    current = _current
    set_network_dtype("float64")
    try:
        return _cast(model.init(rng, x, **kw), jnp.float64)
    finally:
        set_network_dtype(current)


def _apply_low(model, low, variables, x, **kw):
    return model.apply(_cast(variables, low), x, **kw)


def state_kwargs(model):
    """
    MCState kwargs for the network: {"model": model} in float64; otherwise
    init_fun/apply_fun (model=None) keeping float64 parameters around the
    low-precision network. NetKet ignores init_fun/apply_fun when a model is
    given. HashablePartial: one compilation per architecture, as with a model.
    """
    if _current == "float64":
        return {"model": model}
    from netket.jax import HashablePartial

    low = DTYPES[_current][0]
    return {
        "model": None,
        "init_fun": HashablePartial(_init64, model),
        "apply_fun": HashablePartial(_apply_low, model, low),
    }
