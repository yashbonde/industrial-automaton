"""Memory-augmented models with external tape/memory.

Models:
- BabyNTM: Simplified Neural Turing Machine
- TapeRNN: moved to models_torch/tape.py (PyTorch + MPS)

Shared utilities:
- build_op_matrices(): 5 operation matrices (identity, shift-left, shift-right, etc.)
- apply_memory_ops(): Weighted combination of memory operations
"""

from typing import NamedTuple, Any
from pydantic import BaseModel

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models_jax.common import BaseAutomata


# === SHARED MEMORY OPERATIONS ===
# Used by both BabyNTM and TapeRNN

def build_op_matrices(n: int) -> jnp.ndarray:
    """Build (5, N, N) stack of operation matrices for memory size N.

    Operations: [rotate_right, rotate_left, no_op, pop_right, pop_left]
    """
    I = jnp.eye(n)

    rotate_right = jnp.roll(I, 1, axis=0)
    rotate_left = jnp.roll(I, -1, axis=0)
    no_op = I

    # pop_right: shift rows down, zero-fill top row
    # Row i gets row i-1's content; row 0 becomes zero
    pop_right = jnp.zeros((n, n))
    if n > 1:
        pop_right = pop_right.at[1:, :-1].set(jnp.eye(n - 1))

    # pop_left: shift rows up, zero-fill bottom row
    # Row i gets row i+1's content; row N-1 becomes zero
    pop_left = jnp.zeros((n, n))
    if n > 1:
        pop_left = pop_left.at[:-1, 1:].set(jnp.eye(n - 1))

    return jnp.stack([rotate_right, rotate_left, no_op, pop_right, pop_left])


def apply_memory_ops(
    memory: jnp.ndarray,
    action_weights: jnp.ndarray,
    op_matrices: jnp.ndarray,
) -> jnp.ndarray:
    """Apply weighted combination of memory operations.

    Args:
        memory: (N, M) memory matrix
        action_weights: (5,) softmax weights over operations
        op_matrices: (5, N, N) operation matrices

    Returns:
        (N, M) new memory matrix: sum_i a[i] * OP[i] @ M
    """
    # (5, N, N) @ (N, M) -> (5, N, M), then weight by (5,) and sum
    return jnp.einsum("i,ijk,kl->jl", action_weights, op_matrices, memory)


# === VANILLA RNN CELL ===

class VanillaRNNCell(eqx.Module):
    """Simple Elman RNN cell: h_t = tanh(W [x_t; h_{t-1}] + b)"""
    linear: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, *, key):
        self.linear = eqx.nn.Linear(input_size + hidden_size, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(self.linear(jnp.concatenate([x, h])))


# === TAPE-RNN — moved to models_torch/tape.py (PyTorch + MPS) ===
# When model="tape_rnn" the CLI automatically routes to TorchTrainer.
# Kept here (commented) as reference only.

# class TapeRNNState(NamedTuple):
#     memory: jnp.ndarray   # (num_heads, MemorySize, memory_cell_size)
#     hidden: Any           # (hidden_size,) array or tuple of arrays for LSTM
#     pos_tape: jnp.ndarray # (num_heads, MemorySize, pos_dim) - FIXED, only rolled
#
# class TapeRNNConfig(BaseModel):
#     embedding_dim: int = 32
#     hidden_size: int = 256
#     memory_size: int = 40
#     memory_cell_size: int = 8
#     pos_dim: int = 16
#     use_gru: bool = False
#     use_lstm: bool = False
#     num_heads: int = 1
#
# class TapeRNN(BaseAutomata):
#     """Tape-RNN matching Delétang et al. (2023).
#     MOVED to models_torch/tape.py — use TorchTrainer + MPS.
#     Full implementation preserved there verbatim.
#     """
#     ... (see models_torch/tape.py)
