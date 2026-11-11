"""
PCvT: Positional Convolutional Transformer, and SiteHead.

Backbone that merges the two transformer backbones of the project and
removes their weak points (see docs/12_PCvT.md):

    ViT2D   factored attention: learnable circulant weights alpha(r_j - r_i),
            independent of the input (signed, long range, cheap).
    CvT3    convolutional embedding + depthwise convolutions (local context)
            + softmax attention that depends on the input (no positions).

PCvT attention (per head h):

    A_ij = softmax_j( tau_h cos(q_i, k_j) + b_h(r_j - r_i) ) + alpha_h(r_j - r_i)

    * tau cos(q, k)   content term (CvT3) with bounded logits (|.| <= tau_max):
                      the softmax cannot saturate, also in float32.
    * b(r)            relative-position bias: the content attention knows the
                      distance (CvT3 has no positional information in it).
    * alpha(r)        signed circulant weights (ViT2D factored attention).

The tables b and alpha are stored on a fixed window (2R+1)x(2R+1) of
displacements (R = rel_range), so every parameter is independent of the
lattice size: the weights of a 6x6 run can be loaded on 8x8, 10x10...

Block (pre-norm, the residual stream is never normalised or squashed):

    x = x + W_O Attn(LN(x))          W_O and the last FFN layer start small:
    x = x + W_2 gelu(W_1 LN(x))      every block is ~identity at init.

SiteHead: per-site complex log-amplitudes for Sequivariant phase_mode "site",

    psi_k(s) = exp(sum_r u_r) * sum_r chi_k(r) exp(f_r),

the exact momentum projection of an unconstrained state (one forward pass).
The phases start at zero.
"""

import flax.linen as nn
import jax.numpy as jnp
import jax.typing as jt
import numpy as np

from ...toolbox import DepthwiseConv

DTYPE = jnp.float64  # switched by NN.precision.set_network_dtype

ACTIVATIONS = {
    "gelu": nn.gelu,
    "swish": nn.swish,
    "relu": nn.relu,
    "elu": nn.elu,
    "tanh": jnp.tanh,
}

# Output layers of the residual branches: variance = SMALL_INIT / fan_in
# (lecun: 1 / fan_in), i.e. branch output ~ 0.1 x a standard layer at init.
SMALL_INIT = 1e-2


def small_init(scale=SMALL_INIT):
    return nn.initializers.variance_scaling(scale, "fan_in", "truncated_normal")


def relative_index(grid, rel_range):
    """
    (N, N) index into a flattened (2R+1, 2R+1) table of the displacement
    r_j - r_i on a periodic HxW grid. Displacements are folded to
    [-L/2, L/2) and clipped to [-R, R] (beyond R they share the border entry).
    """
    H, W = grid
    R = rel_range
    y, x = np.divmod(np.arange(H * W), W)

    def fold(d, L):
        d = (d + L // 2) % L - L // 2
        return np.clip(d, -R, R) + R

    dy = fold(y[None, :] - y[:, None], H)
    dx = fold(x[None, :] - x[:, None], W)
    return dy * (2 * R + 1) + dx


class RelPosAttention(nn.Module):
    """Multi-head attention with content, relative-position and signed terms."""

    n_heads: int
    kernel: tuple = (3, 3)  # depthwise kernel giving local context to q, k, v
    content: bool = True  # tau cos(q, k) term (False: positional attention only)
    signed: bool = True  # alpha(r) term (False: softmax attention only)
    rel_range: int = 6
    tau_max: float = 20.0

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        B, H, W, C = x.shape
        N, h = H * W, self.n_heads
        if C % h != 0:
            raise ValueError(f"PCvT: channels={C} not divisible by n_heads={h}")
        d = C // h
        dense = lambda name, init=nn.linear.default_kernel_init, bias=False: nn.Dense(
            C, use_bias=bias, dtype=DTYPE, param_dtype=DTYPE, kernel_init=init, name=name
        )

        # local context (CvT3: one depthwise conv per projection; here shared)
        c = x + DepthwiseConv(kernel_size=self.kernel, name="local")(x)

        table = (h, (2 * self.rel_range + 1) ** 2)
        idx = relative_index((H, W), self.rel_range)

        bias = self.param("rel_bias", nn.initializers.zeros_init(), table, DTYPE)
        logits = bias[:, idx][None]  # (1, h, N, N)
        if self.content:
            q = dense("query")(c).reshape(B, N, h, d)
            k = dense("key")(c).reshape(B, N, h, d)
            q = q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            k = k / jnp.sqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            log_tau = self.param("log_tau", nn.initializers.zeros_init(), (h,), DTYPE)
            tau = jnp.exp(jnp.minimum(log_tau, np.log(self.tau_max)))
            logits = logits + tau[None, :, None, None] * jnp.einsum("bihd,bjhd->bhij", q, k)
        attn = nn.softmax(logits, axis=-1)

        if self.signed:
            # same scale for every lattice size: sum_j alpha_ij^2 ~ 1
            alpha = self.param(
                "rel_alpha", nn.initializers.normal(stddev=1.0), table, DTYPE
            )
            attn = attn + alpha[:, idx][None] / np.sqrt(N)

        v = dense("value")(c).reshape(B, N, h, d)
        out = jnp.einsum("bhij,bjhd->bihd", attn, v).reshape(B, H, W, C)
        return dense("out", init=small_init(), bias=True)(out)


class PCvTBlock(nn.Module):
    n_heads: int
    kernel: tuple = (3, 3)
    ffn_factor: int = 2
    ffn_act: str = "gelu"
    content: bool = True
    signed: bool = True
    rel_range: int = 6

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        C = x.shape[-1]
        norm = lambda name: nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name=name)

        x = x + RelPosAttention(
            n_heads=self.n_heads,
            kernel=self.kernel,
            content=self.content,
            signed=self.signed,
            rel_range=self.rel_range,
            name="attn",
        )(norm("norm_attn")(x))

        y = nn.Dense(C * self.ffn_factor, dtype=DTYPE, param_dtype=DTYPE, name="ffn_in")(
            norm("norm_ffn")(x)
        )
        y = ACTIVATIONS[self.ffn_act](y)
        y = nn.Dense(
            C, dtype=DTYPE, param_dtype=DTYPE, kernel_init=small_init(), name="ffn_out"
        )(y)
        return x + y


class PCvT(nn.Module):
    """
    Positional Convolutional Transformer (see the module docstring).

    Per stage i: circular convolutional embedding (kernel[i], strides[i]) ->
    n_blocks[i] PCvTBlocks. A LayerNorm closes the network. Input
    (B, L0, L1, 1); output (B, H, W, channels[-1]), equivariant under
    translations by the total stride (Sequivariant `coset`).
    """

    n_blocks: tuple[int, ...]
    channels: tuple[int, ...]
    n_heads: tuple[int, ...]
    kernel: tuple[tuple[int, int], ...] = ((3, 3),)
    strides: tuple[tuple[int, int], ...] = ((1, 1),)
    ffn_factor: int = 2
    ffn_act: str = "gelu"
    content: bool = True
    signed: bool = True
    rel_range: int = 6

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        n_stages = len(self.n_blocks)
        for name in ("channels", "n_heads", "kernel", "strides"):
            if len(getattr(self, name)) != n_stages:
                raise ValueError(
                    f"PCvT: {name} needs one entry per stage ({n_stages}), "
                    f"got {getattr(self, name)}"
                )
        x = x.astype(DTYPE)  # spins come as float64 (see NN.precision)

        for i in range(n_stages):
            if i > 0:
                x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name=f"norm_stage{i}")(x)
            x = nn.Conv(
                features=self.channels[i],
                kernel_size=tuple(self.kernel[i]),
                strides=tuple(self.strides[i]),
                padding="CIRCULAR",
                dtype=DTYPE,
                param_dtype=DTYPE,
                name=f"embed{i}",
            )(x)
            for j in range(self.n_blocks[i]):
                x = PCvTBlock(
                    n_heads=self.n_heads[i],
                    ffn_factor=self.ffn_factor,
                    ffn_act=self.ffn_act,
                    content=self.content,
                    signed=self.signed,
                    rel_range=self.rel_range,
                    name=f"stage{i}_block{j}",
                )(x)

        return nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE, name="norm_out")(x)


class SiteHead(nn.Module):
    """
    Per-site head for Sequivariant(phase_mode="site").

    Input: equivariant grid (B, H, W, C). Output: (B, H, W) complex,

        F_r = f_r + sum_r' u_r',      (f_r, u_r) = MLP(LN(x_r))

    f_r is the log-amplitude of the unprojected state seen from site r and
    sum u an extensive invariant (Jastrow-like) term (`pooled`). The
    imaginary parts start small but not at zero: a real psi is a saddle point
    of the energy (the phase gradient vanishes by symmetry).
    """

    d_model: int
    pooled: bool = True

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        x = nn.LayerNorm(dtype=DTYPE, param_dtype=DTYPE)(x.astype(DTYPE))
        x = nn.gelu(nn.Dense(self.d_model, dtype=DTYPE, param_dtype=DTYPE, name="hidden")(x))
        n_out = 2 if self.pooled else 1
        re = nn.Dense(
            n_out, dtype=DTYPE, param_dtype=DTYPE, kernel_init=small_init(), name="real"
        )(x)
        im = nn.Dense(
            n_out,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=DTYPE,
            kernel_init=small_init(),  # not zero: a real psi is a saddle (stays real)
            name="imag",
        )(x)
        out = re + 1j * im  # (B, H, W, n_out)
        f = out[..., 0]
        if self.pooled:
            f = f + out[..., 1].sum(axis=(1, 2), keepdims=True)
        return f
