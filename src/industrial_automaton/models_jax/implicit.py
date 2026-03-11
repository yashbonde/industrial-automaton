"""Implicit memory models - sequential processing with hidden state.

Models:
- LSTM: Long Short-Term Memory
- (Future: RNN, GRU)
"""

from typing import NamedTuple
from pydantic import BaseModel, Field
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models_jax.common import BaseAutomata

class LSTMState(NamedTuple):
    hidden: jnp.ndarray  # (stack_height, embedding_dim)
    cell: jnp.ndarray    # (stack_height, embedding_dim)

class LSTMConfig(BaseModel):
    embedding_dim: int = 32
    stack_height: int = Field(default=1, description="Number of stacked LSTMs.")

class LSTM(BaseAutomata):
    """LSTM sequence processor."""
    autoregressive_input: bool = eqx.field(default=False, static=True)

    lstm_cells: list[eqx.nn.LSTMCell]

    vocab_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    stack_height: int = eqx.field(static=True)

    def __init__(self, config: LSTMConfig, *, key):
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = config.embedding_dim
        self.stack_height = config.stack_height
        
        # Split key for multiple layers
        keys = jax.random.split(key, config.stack_height)
        self.lstm_cells = [eqx.nn.LSTMCell(config.embedding_dim, config.embedding_dim, key=k) for k in keys]

    def init_state(self) -> LSTMState:
        return LSTMState(
            hidden=jnp.zeros((self.stack_height, self.embedding_dim)),
            cell=jnp.zeros((self.stack_height, self.embedding_dim)),
        )

    def step(self, x_t, state: LSTMState, is_valid):
        """Single timestep with PAD handling and deep stacking."""
        new_hiddens = []
        new_cells = []
        
        current_inp = x_t
        for i, lstm_cell in enumerate(self.lstm_cells):
            # Each layer takes input from previous layer (or x_t for layer 0)
            # and uses its own hidden/cell state
            h_new, c_new = lstm_cell(current_inp, (state.hidden[i], state.cell[i]))
            
            # Only update if valid (not PAD)
            h = jnp.where(is_valid, h_new, state.hidden[i])
            c = jnp.where(is_valid, c_new, state.cell[i])
            
            new_hiddens.append(h)
            new_cells.append(c)
            
            # Input to next layer is the current layer's hidden state
            current_inp = h
            
        final_state = LSTMState(jnp.stack(new_hiddens), jnp.stack(new_cells))
        return new_hiddens[-1], final_state

    def __call__(self, inputs, state: LSTMState, pad_mask, input_length=None):
        """Process sequence. inputs: (T, embedding_dim). Returns (hidden_states (T, embedding_dim), final_state)."""
        T, D = inputs.shape

        def scan_fn(carry, x_and_valid):
            s = carry
            x_t, is_valid = x_and_valid

            # Always compute (for jit efficiency)
            h, s_new = self.step(x_t, s, is_valid)

            return s_new, h  # Return h for output collection

        init_state = state if state is not None else self.init_state()
        final_state, hiddens = jax.lax.scan(
            scan_fn,
            init_state,
            (inputs, pad_mask)
        )

        return hiddens, final_state  # (T, D), (h, c)

