"""Attention-based models.

Models:
- Transformer: Standard transformer with causal masking
"""

from typing import Optional
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models.common import BaseAutomata

class TransformerConfig(BaseModel):
    embedding_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 128

class MultiheadAttention(eqx.Module):
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim, num_heads, key):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.query_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=False, key=k1)
        self.key_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=False, key=k2)
        self.value_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=False, key=k3)
        self.output_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=False, key=k4)

    def __call__(self, x, mask=None):
        seq_len, embed_dim = x.shape
        q = jax.vmap(self.query_proj)(x).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = jax.vmap(self.key_proj)(x).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = jax.vmap(self.value_proj)(x).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        logits = jnp.einsum('h q d, h k d -> h q k', q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)

        weights = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum('h q k, h k d -> h q d', weights, v)
        attn = attn.transpose(1, 0, 2).reshape(seq_len, embed_dim)
        return jax.vmap(self.output_proj)(attn)

class TransformerBlock(eqx.Module):
    attn: MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, embed_dim, num_heads, key):
        k1, k2 = jax.random.split(key, 2)
        self.attn = MultiheadAttention(embed_dim, num_heads, key=k1)
        self.mlp = eqx.nn.MLP(embed_dim, embed_dim, 4 * embed_dim, 1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.ln2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x, mask=None):
        x = x + self.attn(jax.vmap(self.ln1)(x), mask=mask)
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x

class Transformer(BaseAutomata):
    """Decoder-only Transformer with causal mask."""
    autoregressive_input: bool = eqx.field(default=False, static=True)

    pos_embedding: jnp.ndarray
    blocks: list[TransformerBlock]
    ln_f: eqx.nn.LayerNorm

    vocab_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)

    def __init__(self, config: TransformerConfig, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = config.embedding_dim
        self.max_seq_len = config.max_seq_len
        self.pos_embedding = jax.random.normal(k1, (config.max_seq_len, config.embedding_dim)) * 0.02

        block_keys = jax.random.split(k2, config.num_layers)
        self.blocks = [TransformerBlock(config.embedding_dim, config.num_heads, key=bk) for bk in block_keys]
        self.ln_f = eqx.nn.LayerNorm(config.embedding_dim)

    def __call__(self, inputs, state, pad_mask, input_length=None):
        """Process sequence. inputs: (T, embedding_dim). Returns (hidden_states (T, embedding_dim), final_state)."""
        T, D = inputs.shape
        x = inputs + self.pos_embedding[:T]

        # Create combined mask
        causal_mask = jnp.tril(jnp.ones((T, T)))  # (T, T) lower triangular

        # Expand pad_mask: position i can attend to j if j is not PAD
        pad_mask_2d = pad_mask[None, :] * pad_mask[:, None]  # (T, T)

        # Combine: attend to j if (j <= i) AND (j is not PAD)
        mask = causal_mask * pad_mask_2d  # (T, T)

        for block in self.blocks:
            x = block(x, mask=mask)

        hidden_states = jax.vmap(self.ln_f)(x)
        return hidden_states, state  # Return state for API consistency

    def init_state(self):
        return None
