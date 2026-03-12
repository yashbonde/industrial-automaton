"""Stack-based neural automata.

Models:
- SuzgunStackRNN: Differentiable stack with push/pop operations
"""

from typing import NamedTuple
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models_jax.common import BaseAutomata
from industrial_automaton.models_jax.tape import build_op_matrices, apply_memory_ops

class StackRNNState(NamedTuple):
    stack: jnp.ndarray    # (Depth, Dim)
    hidden: jnp.ndarray   # (Hidden,)

class SuzgunStackRNNConfig(BaseModel):
    embedding_dim: int = 32
    stack_depth: int = 50
    value_dim: int = 1

class SuzgunStackRNN(BaseAutomata):
    """Suzgun Stack-RNN: RNN controller + differentiable stack.

    Architecture (Suzgun et al. 2019):
    h_tilde_{t-1} = h_{t-1} + W_sh * stack_{t-1}[0]
    h_t = tanh(W_ih * x_t + b_ih + W_hh * h_tilde_{t-1} + b_hh)
    a_t = softmax(W_a * h_t)  # [push, pop, no-op]
    n_t = sigmoid(W_n * h_t)  # value to push
    """
    autoregressive_input: bool = eqx.field(default=True, static=True)

    W_ih: jnp.ndarray
    b_ih: jnp.ndarray
    W_hh: jnp.ndarray
    b_hh: jnp.ndarray
    W_sh: jnp.ndarray
    W_a: jnp.ndarray
    W_n: jnp.ndarray

    embedding_dim: int = eqx.field(static=True)
    stack_depth: int = eqx.field(static=True)
    value_dim: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    def __init__(
        self,
        config: SuzgunStackRNNConfig,
        *,
        key,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = config.embedding_dim
        self.stack_depth = config.stack_depth
        self.value_dim = config.value_dim

        scale = 1.0 / (config.embedding_dim**0.5)
        self.W_ih = jax.random.uniform(k1, (config.embedding_dim, config.embedding_dim), minval=-scale, maxval=scale)

        self.b_ih = jnp.zeros(config.embedding_dim)
        self.W_hh = jax.random.uniform(k2, (config.embedding_dim, config.embedding_dim), minval=-scale, maxval=scale)
        self.b_hh = jnp.zeros(config.embedding_dim)

        self.W_sh = jax.random.normal(k3, (config.embedding_dim, config.value_dim)) * 0.1
        self.W_a = jax.random.normal(k4, (3, config.embedding_dim)) * 0.1
        self.W_n = jax.random.normal(k5, (config.value_dim, config.embedding_dim)) * 0.1

    def init_state(self) -> StackRNNState:
        return StackRNNState(
            stack=jnp.zeros((self.stack_depth, self.value_dim)),
            hidden=jnp.zeros(self.embedding_dim),
        )

    def step(self, x_t: jnp.ndarray, state: StackRNNState, is_valid):
        """Single timestep with PAD handling. x_t: (embedding_dim,). Returns (hidden_t, new_state)."""
        stack, hidden = state

        # 1. Inject stack top into hidden state (Suzgun style: additive)
        h_tilde = hidden + self.W_sh @ stack[0]

        # 2. Update hidden state (RNN)
        hidden_new = jnp.tanh(self.W_ih @ x_t + self.b_ih + self.W_hh @ h_tilde + self.b_hh)

        # 3. Action weights (PUSH, POP, NO-OP)
        a_t = jax.nn.softmax(self.W_a @ hidden_new)
        push_weight = a_t[0]
        pop_weight = a_t[1]
        noop_weight = a_t[2]

        # 4. Value to push
        n_t = jax.nn.sigmoid(self.W_n @ hidden_new)

        # 5. Update stack (differentiable)
        # s_t[0] = push * n_t + pop * stack[1] + noop * stack[0]
        # s_t[i] = push * stack[i-1] + pop * stack[i+1] + noop * stack[i]

        stack_push = jnp.concatenate([n_t[None], stack[:-1]], axis=0)
        stack_pop = jnp.concatenate([stack[1:], jnp.zeros((1, self.value_dim))], axis=0)
        stack_noop = stack

        stack_new = push_weight * stack_push + pop_weight * stack_pop + noop_weight * stack_noop

        # Freeze stack operations if PAD
        frozen_state = StackRNNState(
            stack=jnp.where(is_valid, stack_new, stack),
            hidden=jnp.where(is_valid, hidden_new, hidden),
        )

        return hidden_new, frozen_state

    def __call__(self, inputs: jnp.ndarray, state: StackRNNState, pad_mask, input_length=None):
        """Process sequence. inputs: (T, embedding_dim). Returns (hidden_states (T, embedding_dim), final_state)."""
        def scan_fn(carry, x_and_valid):
            current_state = carry
            x_t, is_valid = x_and_valid

            # Always compute (for jit efficiency)
            h_t, new_state = self.step(x_t, current_state, is_valid)

            return new_state, h_t

        init_state = state if state is not None else self.init_state()
        final_state, hidden_states = jax.lax.scan(
            scan_fn,
            init_state,
            (inputs, pad_mask)
        )

        return hidden_states, final_state


# === BABY-NTM ===

class BabyNTMState(NamedTuple):
    memory: jnp.ndarray   # (N, M)
    hidden: jnp.ndarray   # (H,)
    cell: jnp.ndarray     # (H,)

class BabyNTMModelConfig(BaseModel):
    embedding_dim: int = 32
    memory_size: int = 104
    memory_dim: int = 1

class BabyNTM(BaseAutomata):
    """Baby-NTM: LSTM controller + fixed-size memory with 5 deterministic ops."""
    autoregressive_input: bool = eqx.field(default=True, static=True)

    lstm: eqx.nn.LSTMCell
    W_m: jnp.ndarray   # (D, M) — memory read projection
    W_a: jnp.ndarray   # (5, D) — action logits
    W_n: jnp.ndarray   # (M, D) — new memory value
    W_g: jnp.ndarray   # (M, D) — erase gate projection
    op_matrices: jnp.ndarray  # (5, N, N) — static

    vocab_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
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
        self.embedding_dim = config.embedding_dim
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim

        # LSTM: embedding_dim -> embedding_dim
        self.lstm = eqx.nn.LSTMCell(config.embedding_dim, config.embedding_dim, key=k1)

        scale = 0.1
        self.W_m = jax.random.normal(k2, (config.embedding_dim, config.memory_dim)) * scale
        self.W_a = jax.random.normal(k3, (5, config.embedding_dim)) * scale
        self.W_n = jax.random.normal(k4, (config.memory_dim, config.embedding_dim)) * scale
        self.W_g = jax.random.normal(k5, (config.memory_dim, config.embedding_dim)) * scale

        self.op_matrices = build_op_matrices(config.memory_size)

    def init_state(self) -> BabyNTMState:
        return BabyNTMState(
            memory=jnp.zeros((self.memory_size, self.memory_dim)),
            hidden=jnp.zeros(self.embedding_dim),
            cell=jnp.zeros(self.embedding_dim),
        )

    def step(self, x_t: jnp.ndarray, state: BabyNTMState, is_valid):
        """Single timestep with PAD handling. x_t: (embedding_dim,). Returns (hidden_t, new_state)."""
        memory, hidden, cell = state

        # 1. Augment hidden with memory read (first entry)
        h_tilde = hidden + self.W_m @ memory[0]

        # 2. LSTM step
        hidden_new, cell_new = self.lstm(x_t, (h_tilde, cell))

        # 3. Action weights
        action_weights = jax.nn.softmax(self.W_a @ hidden_new)

        # 4. New memory value and erase gate
        n_t = jax.nn.sigmoid(self.W_n @ hidden_new)
        g_t = jax.nn.sigmoid(self.W_g @ hidden_new)

        # 5. Apply memory operations
        memory_new = apply_memory_ops(memory, action_weights, self.op_matrices)

        # 6. Gated write: memory = (1 - g_t) * memory_old + g_t * n_t
        # This bounds memory values to prevent NaN divergence
        memory_new = memory_new.at[0].set((1 - g_t) * memory_new[0] + g_t * n_t)

        # Freeze memory/weights if PAD
        # Only update state if is_valid is True, otherwise keep old state
        frozen_state = BabyNTMState(
            memory=jnp.where(is_valid, memory_new, memory),
            hidden=jnp.where(is_valid, hidden_new, hidden),
            cell=jnp.where(is_valid, cell_new, cell),
        )

        return hidden_new, frozen_state

    def __call__(self, inputs: jnp.ndarray, state: BabyNTMState, pad_mask, input_length=None):
        """Process sequence. inputs: (T, embedding_dim). Returns (hidden_states (T, embedding_dim), final_state)."""

        def scan_fn(carry, x_and_valid):
            current_state = carry
            x_t, is_valid = x_and_valid

            # Always compute (for jit efficiency)
            h_t, new_state = self.step(x_t, current_state, is_valid)

            return new_state, h_t

        init_state = state if state is not None else self.init_state()
        final_state, hidden_states = jax.lax.scan(
            scan_fn,
            init_state,
            (inputs, pad_mask)
        )

        return hidden_states, final_state

