"""Memory-augmented models with external tape/memory.

Models:
- BabyNTM: Simplified Neural Turing Machine
- TapeRNN: RNN with tape-based external memory

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
    memory: jnp.ndarray   # (num_heads, MemorySize, memory_cell_size)
    hidden: Any           # (hidden_size,) array or tuple of arrays for LSTM

class TapeRNNConfig(BaseModel):
    embedding_dim: int = 32
    hidden_size: int = 256          # RNN hidden size (independent of embedding_dim)
    memory_size: int = 40           # Number of tape cells
    memory_cell_size: int = 8       # Dimension of each tape cell
    use_gru: bool = False           # Use GRU instead of VanillaRNN
    use_lstm: bool = False          # Use LSTM instead of VanillaRNN
    num_heads: int = 1              # Number of independent tape heads

class TapeRNN(BaseAutomata):
    """Tape-RNN matching Delétang et al. (2023) with enhancements:
    - Flexible controller (VanillaRNN, GRU, or LSTM)
    - Windowed read head (3 cells) per head
    - MLP write head (shared or independent per head via single large linear)
    - Learnable tape initialization
    - Initial action bias favoring Right movement
    - Multi-head support: multiple independent tape positions
    - 5 tape actions per head: Stay, Left, Right, JumpLeft(L), JumpRight(L)
    - hidden_size decoupled from embedding_dim
    """
    rnn: eqx.Module                 # eqx.nn.GRUCell, eqx.nn.LSTMCell, or VanillaRNNCell
    W_m: jnp.ndarray               # (hidden_size, num_heads * memory_cell_size * 3)
    W_a: jnp.ndarray               # (num_heads * 5, hidden_size) — action logits
    b_a: jnp.ndarray               # (num_heads * 5,) — action bias
    write_l1: eqx.nn.Linear        # hidden_size -> 64
    write_l2: eqx.nn.Linear        # 64 -> 64
    write_l3: eqx.nn.Linear        # 64 -> num_heads * memory_cell_size
    tape_init: jnp.ndarray         # (memory_size, memory_cell_size)

    vocab_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    memory_cell_size: int = eqx.field(static=True)
    use_gru: bool = eqx.field(static=True)
    use_lstm: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(self, config: TapeRNNConfig, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        self.memory_cell_size = config.memory_cell_size
        self.use_gru = config.use_gru
        self.use_lstm = config.use_lstm
        self.num_heads = config.num_heads

        # RNN input = embedding + memory read projection
        rnn_input_size = config.embedding_dim + config.hidden_size
        if self.use_lstm:
            self.rnn = eqx.nn.LSTMCell(rnn_input_size, config.hidden_size, key=k1)
        elif self.use_gru:
            self.rnn = eqx.nn.GRUCell(rnn_input_size, config.hidden_size, key=k1)
        else:
            self.rnn = VanillaRNNCell(rnn_input_size, config.hidden_size, key=k1)

        scale = 0.1
        # Windowed read: read current cell and its immediate neighbors for each head
        self.W_m = jax.random.normal(k2, (config.hidden_size, config.num_heads * config.memory_cell_size * 3)) * scale
        self.W_a = jax.random.normal(k3, (config.num_heads * 5, config.hidden_size)) * scale
        
        # Initial action bias: strongly favor Right movement for all heads
        bias = np.zeros(config.num_heads * 5)
        for h in range(config.num_heads):
            bias[h * 5 + 2] = 3.0
        self.b_a = jnp.array(bias)
        
        # MLP write head: hidden_size -> 64 -> 64 -> num_heads * memory_cell_size
        k4a, k4b, k4c = jax.random.split(k4, 3)
        self.write_l1 = eqx.nn.Linear(config.hidden_size, 64, key=k4a)
        self.write_l2 = eqx.nn.Linear(64, 64, key=k4b)
        self.write_l3 = eqx.nn.Linear(64, config.num_heads * config.memory_cell_size, key=k4c)
        
        # Learnable tape initialization with sinusoidal positional encoding
        pe = np.zeros((config.memory_size, config.memory_cell_size))
        position = np.arange(config.memory_size)[:, np.newaxis]
        div_term = np.exp(np.arange(0, config.memory_cell_size, 2) * -(np.log(10000.0) / config.memory_cell_size))
        pe[:, 0::2] = np.sin(position * div_term)
        if config.memory_cell_size % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        self.tape_init = jnp.array(pe) + jax.random.normal(k5, (config.memory_size, config.memory_cell_size)) * scale

    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def init_state(self) -> TapeRNNState:
        dtype = self.tape_init.dtype
        if self.use_lstm:
            hidden = (jnp.zeros(self.hidden_size, dtype=dtype), jnp.zeros(self.hidden_size, dtype=dtype))
        else:
            hidden = jnp.zeros(self.hidden_size, dtype=dtype)
        
        return TapeRNNState(
            memory=jnp.tile(self.tape_init[None, ...], (self.num_heads, 1, 1)),
            hidden=hidden,
        )

    def step(self, x_t: jnp.ndarray, state: TapeRNNState, jump_len: int, is_valid):
        """Single timestep. x_t: (embedding_dim,). Returns (hidden_t, new_state)."""
        memory, hidden = state # memory: (num_heads, memory_size, memory_cell_size)
        
        # h_t is the actual hidden state vector used for attention/output
        h_t = hidden[0] if self.use_lstm else hidden

        # 1. Windowed Read for each head
        windows = []
        for h in range(self.num_heads):
            # Each head sees its own shifted view of the tape
            m_h = memory[h]
            # Window of size 3: left, center, right
            w = jnp.concatenate([m_h[-1], m_h[0], m_h[1]])
            windows.append(w)
        
        mem_read = self.W_m @ jnp.concatenate(windows)   # (hidden_size,)

        # 2. Concatenate embedding + memory read, then step RNN
        rnn_input = jnp.concatenate([x_t, mem_read])  # (embedding_dim + hidden_size,)
        hidden_new = self.rnn(rnn_input, hidden)       # (hidden_size,) or tuple

        h_new_t = hidden_new[0] if self.use_lstm else hidden_new

        # 3. Action weights over 5 tape ops for each head
        action_logits = self.W_a @ h_new_t + self.b_a
        a_t = jax.nn.softmax(action_logits.reshape(self.num_heads, 5), axis=-1)

        # 4. Write value via MLP for each head
        n_t_all = jax.nn.relu(self.write_l1(h_new_t))
        n_t_all = jax.nn.relu(self.write_l2(n_t_all))
        n_t_all = self.write_l3(n_t_all) # (num_heads * memory_cell_size,)
        n_t_all = n_t_all.reshape(self.num_heads, self.memory_cell_size)

        memory_new_list = []
        for h in range(self.num_heads):
            n_t = n_t_all[h]
            
            # Write to tape cell 0, then apply weighted shift
            m_h = memory[h]
            memory_w = m_h.at[0].set(n_t)

            # Optimized shift using jnp.roll directly
            m_h_new = (
                a_t[h, 0] * memory_w +
                a_t[h, 1] * jnp.roll(memory_w, shift=1, axis=0) +
                a_t[h, 2] * jnp.roll(memory_w, shift=-1, axis=0) +
                a_t[h, 3] * jnp.roll(memory_w, shift=jump_len, axis=0) +
                a_t[h, 4] * jnp.roll(memory_w, shift=-jump_len, axis=0)
            )
            memory_new_list.append(m_h_new)

        memory_new = jnp.stack(memory_new_list)

        # Handle freezing correctly for tuples
        if self.use_lstm:
            hidden_frozen = (
                jnp.where(is_valid, hidden_new[0], hidden[0]),
                jnp.where(is_valid, hidden_new[1], hidden[1])
            )
        else:
            hidden_frozen = jnp.where(is_valid, hidden_new, hidden)

        frozen_state = TapeRNNState(
            memory=jnp.where(is_valid[..., None, None], memory_new, memory),
            hidden=hidden_frozen,
        )
        return h_new_t, frozen_state

    def __call__(self, inputs: jnp.ndarray, state: TapeRNNState, pad_mask, input_length=None):
        """Process sequence. inputs: (T, embedding_dim). Returns (hidden_states (T, hidden_size), final_state)."""
        # Use actual input length for jumps; fall back to full seq len.
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
