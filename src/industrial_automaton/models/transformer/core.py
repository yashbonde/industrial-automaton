"""Vanilla GPT-style Transformer implementation (Equinox)."""

from typing import Optional
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE

class TransformerConfig(BaseModel):
    embed_dim: int = 32
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
        self.query_proj = eqx.nn.Linear(embed_dim, embed_dim, key=k1)
        self.key_proj = eqx.nn.Linear(embed_dim, embed_dim, key=k2)
        self.value_proj = eqx.nn.Linear(embed_dim, embed_dim, key=k3)
        self.output_proj = eqx.nn.Linear(embed_dim, embed_dim, key=k4)

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

class Transformer(eqx.Module):
    """Decoder-only Transformer with causal mask."""
    embedding: eqx.nn.Linear
    pos_embedding: jnp.ndarray
    blocks: list[TransformerBlock]
    ln_f: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    
    vocab_size: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)

    def __init__(self, config: TransformerConfig, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.vocab_size = VOCAB_SIZE
        self.max_seq_len = config.max_seq_len
        self.embedding = eqx.nn.Linear(VOCAB_SIZE, config.embed_dim, key=k1)
        self.pos_embedding = jax.random.normal(k2, (config.max_seq_len, config.embed_dim)) * 0.02

        block_keys = jax.random.split(k3, config.num_layers)
        self.blocks = [TransformerBlock(config.embed_dim, config.num_heads, key=bk) for bk in block_keys]
        self.ln_f = eqx.nn.LayerNorm(config.embed_dim)
        self.head = eqx.nn.Linear(config.embed_dim, VOCAB_SIZE, key=key)

    def __call__(self, inputs, state=None):
        # inputs: (T, vocab_size) one-hot
        T, V = inputs.shape
        x = jax.vmap(self.embedding)(inputs)
        x = x + self.pos_embedding[:T]
        
        mask = jnp.tril(jnp.ones((T, T)))
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = jax.vmap(self.ln_f)(x)
        logits = jax.vmap(self.head)(x)
        return logits, state # Return state for API consistency
    
    def init_state(self):
        return None
