"""Baby Neural Turing Machine (Suzgun et al. 2019, Section 3.3)."""

from typing import NamedTuple
from pydantic import BaseModel

import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from .memory_ops import build_op_matrices, apply_memory_ops


class BabyNTMState(NamedTuple):
    memory: jnp.ndarray   # (N, M)
    hidden: jnp.ndarray   # (H,)
    cell: jnp.ndarray     # (H,)

class BabyNTMModelConfig(BaseModel):
    hidden_size: int = 8
    memory_size: int = 104
    memory_dim: int = 1


class BabyNTM(eqx.Module):
    """Baby-NTM: LSTM controller + fixed-size memory with 5 deterministic ops."""

    lstm: eqx.nn.LSTMCell
    W_m: jnp.ndarray   # (H, M) — memory read projection
    W_y: jnp.ndarray   # (vocab, H) — output projection
    W_a: jnp.ndarray   # (5, H) — action logits
    W_n: jnp.ndarray   # (M, H) — new memory value
    op_matrices: jnp.ndarray  # (5, N, N) — static

    vocab_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    memory_dim: int = eqx.field(static=True)

    def __init__(
        self,
        config: BabyNTMModelConfig,
        *,
        key,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.vocab_size = VOCAB_SIZE
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim

        # LSTM input = vocab_size (one-hot)
        self.lstm = eqx.nn.LSTMCell(VOCAB_SIZE, config.hidden_size, key=k1)

        scale = 0.1
        self.W_m = jax.random.normal(k2, (config.hidden_size, config.memory_dim)) * scale
        self.W_y = jax.random.normal(k3, (VOCAB_SIZE, config.hidden_size)) * scale
        self.W_a = jax.random.normal(k4, (5, config.hidden_size)) * scale
        self.W_n = jax.random.normal(k5, (config.memory_dim, config.hidden_size)) * scale

        self.op_matrices = build_op_matrices(config.memory_size)

    def init_state(self) -> BabyNTMState:
        return BabyNTMState(
            memory=jnp.zeros((self.memory_size, self.memory_dim)),
            hidden=jnp.zeros(self.hidden_size),
            cell=jnp.zeros(self.hidden_size),
        )

    def step(self, x_t: jnp.ndarray, state: BabyNTMState):
        """Single timestep. x_t: (vocab,) one-hot. Returns (y_t, new_state)."""
        memory, hidden, cell = state

        # 1. Augment hidden with memory read (first entry)
        h_tilde = hidden + self.W_m @ memory[0]

        # 2. LSTM step
        hidden_new, cell_new = self.lstm(x_t, (h_tilde, cell))

        # 3. Output
        y_t = jax.nn.sigmoid(self.W_y @ hidden_new)

        # 4. Action weights
        action_weights = jax.nn.softmax(self.W_a @ hidden_new)

        # 5. New memory value
        n_t = jax.nn.sigmoid(self.W_n @ hidden_new)

        # 6. Apply memory operations
        memory_new = apply_memory_ops(memory, action_weights, self.op_matrices)

        # 7. Write new value to first entry
        memory_new = memory_new.at[0].add(n_t)

        new_state = BabyNTMState(memory_new, hidden_new, cell_new)
        return y_t, new_state

    def __call__(self, inputs: jnp.ndarray, state: BabyNTMState):
        """Process sequence. inputs: (T, vocab) one-hot. Returns (outputs (T, vocab), final_state)."""

        def scan_fn(state, x_t):
            y_t, new_state = self.step(x_t, state)
            return new_state, y_t

        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        return outputs, final_state
