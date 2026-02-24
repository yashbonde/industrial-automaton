"""Standard LSTM implementation (Equinox)."""

from typing import NamedTuple
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE

class LSTMState(NamedTuple):
    hidden: jnp.ndarray
    cell: jnp.ndarray

class LSTMConfig(BaseModel):
    hidden_size: int = 32

class LSTM(eqx.Module):
    """LSTM sequence processor."""
    lstm: eqx.nn.LSTMCell
    head: eqx.nn.Linear

    vocab_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, config: LSTMConfig, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.vocab_size = VOCAB_SIZE
        self.hidden_size = config.hidden_size
        self.lstm = eqx.nn.LSTMCell(VOCAB_SIZE, config.hidden_size, key=k1)
        self.head = eqx.nn.Linear(config.hidden_size, VOCAB_SIZE, key=k2)

    def init_state(self) -> LSTMState:
        return LSTMState(
            hidden=jnp.zeros(self.hidden_size),
            cell=jnp.zeros(self.hidden_size),
        )

    def step(self, x_t, state: LSTMState):
        h_new, c_new = self.lstm(x_t, (state.hidden, state.cell))
        y_t = self.head(h_new)
        return y_t, LSTMState(h_new, c_new)

    def __call__(self, inputs, state: LSTMState):
        def scan_fn(s, x):
            y, s_new = self.step(x, s)
            return s_new, y
        
        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        return outputs, final_state
