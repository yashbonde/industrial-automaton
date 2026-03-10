"""Memory-augmented models with external tape/memory.

Models:
- BabyNTM: Simplified Neural Turing Machine
- TapeRNN: RNN with tape-based external memory

Shared utilities:
- build_op_matrices(): 5 operation matrices (identity, shift-left, shift-right, etc.)
- apply_memory_ops(): Weighted combination of memory operations
"""

from typing import NamedTuple
from pydantic import BaseModel

import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models.common import BaseAutomata


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


# === TAPE-RNN ===

class TapeRNNState(NamedTuple):
    memory: jnp.ndarray   # (MemorySize, memory_cell_size)
    hidden: jnp.ndarray   # (hidden_size,)

class TapeRNNConfig(BaseModel):
    embedding_dim: int = 32
    hidden_size: int = 256          # RNN hidden size (independent of embedding_dim)
    memory_size: int = 40           # Number of tape cells
    memory_cell_size: int = 8       # Dimension of each tape cell
    use_gru: bool = False           # Use GRU instead of VanillaRNN

class TapeRNN(BaseAutomata):
    """Tape-RNN matching Delétang et al. (2023):
    - VanillaRNN controller (not LSTM) - GRU optional
    - Memory read concatenated to input (not injected into hidden)
    - MLP write head
    - 5 tape actions: Stay, Left, Right, JumpLeft(L), JumpRight(L)
    - hidden_size decoupled from embedding_dim
    """
    autoregressive_input: bool = eqx.field(default=True, static=True)

    rnn: eqx.Module                 # eqx.nn.GRUCell or VanillaRNNCell
    W_m: jnp.ndarray               # (hidden_size, memory_cell_size) — memory read projection
    W_a: jnp.ndarray               # (5, hidden_size) — action logits
    write_l1: eqx.nn.Linear        # hidden_size -> 64
    write_l2: eqx.nn.Linear        # 64 -> 64
    write_l3: eqx.nn.Linear        # 64 -> memory_cell_size

    vocab_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    memory_cell_size: int = eqx.field(static=True)
    use_gru: bool = eqx.field(static=True)

    def __init__(self, config: TapeRNNConfig, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        self.memory_cell_size = config.memory_cell_size
        self.use_gru = config.use_gru

        # RNN input = embedding + memory read projection (both hidden_size dim)
        rnn_input_size = config.embedding_dim + config.hidden_size
        if self.use_gru:
            self.rnn = eqx.nn.GRUCell(rnn_input_size, config.hidden_size, key=k1)
        else:
            self.rnn = VanillaRNNCell(rnn_input_size, config.hidden_size, key=k1)

        scale = 0.1
        self.W_m = jax.random.normal(k2, (config.hidden_size, config.memory_cell_size)) * scale
        self.W_a = jax.random.normal(k3, (5, config.hidden_size)) * scale
        # MLP: hidden_size -> 64 -> 64 -> memory_cell_size (matching paper)
        k4a, k4b, k4c = jax.random.split(k4, 3)
        self.write_l1 = eqx.nn.Linear(config.hidden_size, 64, key=k4a)
        self.write_l2 = eqx.nn.Linear(64, 64, key=k4b)
        self.write_l3 = eqx.nn.Linear(64, config.memory_cell_size, key=k4c)

    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def init_state(self) -> TapeRNNState:
        return TapeRNNState(
            memory=jnp.zeros((self.memory_size, self.memory_cell_size)),
            hidden=jnp.zeros(self.hidden_size),
        )

    def step(self, x_t: jnp.ndarray, state: TapeRNNState, jump_len: int, is_valid):
        """Single timestep. x_t: (embedding_dim,). Returns (hidden_t, new_state)."""
        memory, hidden = state

        # 1. Read current tape cell and project to hidden_size
        mem_read = self.W_m @ memory[0]   # (hidden_size,)

        # 2. Concatenate embedding + memory read, then step RNN
        rnn_input = jnp.concatenate([x_t, mem_read])  # (embedding_dim + hidden_size,)
        hidden_new = self.rnn(rnn_input, hidden)       # (hidden_size,)

        # 3. Action weights over 5 tape ops
        a_t = jax.nn.softmax(self.W_a @ hidden_new)

        # 4. Write value via MLP
        n_t = jax.nn.relu(self.write_l1(hidden_new))
        n_t = jax.nn.relu(self.write_l2(n_t))
        n_t = self.write_l3(n_t)  # (memory_cell_size,)

        # 5. Write to tape cell 0, then apply weighted shift
        memory_w = memory.at[0].set(n_t)

        eye = jnp.eye(self.memory_size)
        ops = jnp.stack([
            eye,                                          # Stay
            jnp.roll(eye, shift=1, axis=0),              # Left
            jnp.roll(eye, shift=-1, axis=0),             # Right
            jnp.roll(eye, shift=jump_len, axis=0),       # JumpLeft(L)
            jnp.roll(eye, shift=-jump_len, axis=0),      # JumpRight(L)
        ])
        memory_new = jnp.einsum('i,isj,jk->sk', a_t, ops, memory_w)

        frozen_state = TapeRNNState(
            memory=jnp.where(is_valid, memory_new, memory),
            hidden=jnp.where(is_valid, hidden_new, hidden),
        )
        return hidden_new, frozen_state

    def __call__(self, inputs: jnp.ndarray, state: TapeRNNState, pad_mask, input_length=None):
        """Process sequence. inputs: (T, embedding_dim). Returns (hidden_states (T, hidden_size), final_state)."""
        # Use actual input length for jumps; fall back to full seq len.
        # Keep as JAX array — jnp.roll supports traced shift values.
        jump_len = input_length if input_length is not None else jnp.array(inputs.shape[0])

        def scan_fn(carry, x_and_valid):
            current_state = carry
            x_t, is_valid = x_and_valid
            h, new_state = self.step(x_t, current_state, jump_len=jump_len, is_valid=is_valid)
            return new_state, h

        init_state = state if state is not None else self.init_state()
        final_state, hidden_states = jax.lax.scan(
            scan_fn,
            init_state,
            (inputs, pad_mask)
        )
        return hidden_states, final_state
