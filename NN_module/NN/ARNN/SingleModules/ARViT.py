from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

DTYPE = jnp.float64


class CausalTransformerBlock(nn.Module):
    n_heads: int
    n_ffn_layers: int
    embedding_d: int
    param_dtype: Any = DTYPE

    @nn.compact
    def __call__(self, x, mask):
        # Multi-Head Attention pasándole la máscara causal
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d,
            param_dtype=self.param_dtype,
        )(x, x, mask=mask)

        x = x + attn_out
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)
        # Feed-Forward Network (MLP)
        ffn = x
        for _ in range(self.n_ffn_layers):
            ffn = nn.Dense(features=self.embedding_d * 2, param_dtype=self.param_dtype)(
                ffn
            )
            ffn = nn.gelu(ffn)
        ffn = nn.Dense(features=self.embedding_d, param_dtype=self.param_dtype)(ffn)
        x = x + ffn
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)
        return x


class ARViT(nk.models.AbstractARNN):
    hilbert: nk.hilbert
    embedding_d: int = 8
    n_heads: int = 2
    n_blocks: int = 2
    n_ffn_layers: int = 1
    machine_pow: int = 2
    param_dtype: Any = DTYPE

    @nn.compact
    def conditionals_log_psi(self, inputs):

        B, N = inputs.shape

        # 1. SHIFT CAUSAL
        zeros = jnp.zeros((B, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(self.param_dtype)[..., None]
        # 2. EMBEDDING INICIAL
        x = nn.Dense(
            features=self.embedding_d, name="embed", param_dtype=self.param_dtype
        )(x)

        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.01),
            (N, self.embedding_d),
            self.param_dtype,
        )
        x = x + pos_emb
        # 3. MÁSCARA CAUSAL
        mask = nn.make_causal_mask(jnp.empty((B, N)))

        # 4. PASO POR LOS BLOQUES TRANSFORMER
        for i in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d,
                param_dtype=self.param_dtype,
                name=f"causal_block_{i}",
            )(x, mask)
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)

        logits = nn.Dense(
            features=2,
            name="final_dense",
            kernel_init=nn.initializers.zeros,
            param_dtype=self.param_dtype,
        )(x)

        # 6. CONVERSIÓN A LOG-AMPLITUDES CUÁNTICAS
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # real
        phase = nn.Dense(
            2,
            name="phase_dense",
            kernel_init=nn.initializers.normal(0.05),  # NO zeros
            param_dtype=self.param_dtype,
        )(
            x
        )  # real

        return jax.lax.complex(0.5 * log_probs, phase)
