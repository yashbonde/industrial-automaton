"""Tape-RNN implementation (Deletang et al. 2023).
Basically a Turing Machine with a differentiable tape.
"""

from typing import NamedTuple
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE

class TapeRNNState(NamedTuple):
    memory: jnp.ndarray   # (MemorySize, Dim)
    hidden: jnp.ndarray   # (Hidden,)
    cell: jnp.ndarray     # (Hidden,)

class TapeRNNConfig(BaseModel):
    hidden_size: int = 16
    memory_size: int = 50
    memory_dim: int = 1

class TapeRNN(eqx.Module):
    """Tape-RNN: LSTM controller + differentiable tape with 5 actions.
    Actions: [Stay, Left, Right, JumpLeft(L), JumpRight(L)]
    """
    
    lstm: eqx.nn.LSTMCell
    W_m: jnp.ndarray   # (H, M) — memory read projection
    W_y: jnp.ndarray   # (vocab, H) — output projection
    W_a: jnp.ndarray   # (5, H) — action logits
    W_n: jnp.ndarray   # (M, H) — new memory value
    
    vocab_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    memory_dim: int = eqx.field(static=True)
    
    def __init__(
        self,
        config: TapeRNNConfig,
        *,
        key,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.vocab_size = VOCAB_SIZE
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim

        self.lstm = eqx.nn.LSTMCell(VOCAB_SIZE, config.hidden_size, key=k1)

        scale = 0.1
        self.W_m = jax.random.normal(k2, (config.hidden_size, config.memory_dim)) * scale
        self.W_y = jax.random.normal(k3, (VOCAB_SIZE, config.hidden_size)) * scale
        self.W_a = jax.random.normal(k4, (5, config.hidden_size)) * scale
        self.W_n = jax.random.normal(k5, (config.memory_dim, config.hidden_size)) * scale

    def init_state(self) -> TapeRNNState:
        return TapeRNNState(
            memory=jnp.zeros((self.memory_size, self.memory_dim)),
            hidden=jnp.zeros(self.hidden_size),
            cell=jnp.zeros(self.hidden_size),
        )

    def step(self, x_t: jnp.ndarray, state: TapeRNNState, jump_len: int = 0):
        memory, hidden, cell = state
        
        # 1. Read from tape (at position 0, we move the tape instead of a head)
        h_tilde = hidden + self.W_m @ memory[0]
        
        # 2. LSTM step
        hidden_new, cell_new = self.lstm(x_t, (h_tilde, cell))
        
        # 3. Output
        y_t = self.W_y @ hidden_new
        
        # 4. Action weights
        a_t = jax.nn.softmax(self.W_a @ hidden_new)
        
        # 5. New memory value to write at current position (0)
        n_t = jax.nn.sigmoid(self.W_n @ hidden_new)
        
        # 6. Apply write
        memory_w = memory.at[0].set(n_t)
        
        # 7. Apply tape shift (equivalent to moving head)
        # Actions: 0:Stay, 1:Left, 2:Right, 3:JumpLeft, 4:JumpRight
        # Shifting memory right = head moves left
        # Shifting memory left = head moves right
        
        eye = jnp.eye(self.memory_size)
        op_stay = eye
        op_left = jnp.roll(eye, shift=1, axis=0)
        op_right = jnp.roll(eye, shift=-1, axis=0)
        op_jleft = jnp.roll(eye, shift=jump_len, axis=0)
        op_jright = jnp.roll(eye, shift=-jump_len, axis=0)
        
        ops = jnp.stack([op_stay, op_left, op_right, op_jleft, op_jright])
        
        # memory_new = sum(a_i * (op_i @ memory_w))
        # ops: (5, S, S), memory_w: (S, M), a_t: (5,)
        # result: (S, M)
        memory_new = jnp.einsum('i,isj,jk->sk', a_t, ops, memory_w)
        
        return y_t, TapeRNNState(memory_new, hidden_new, cell_new)

    def __call__(self, inputs: jnp.ndarray, state: TapeRNNState):
        jump_len = inputs.shape[0]
        def scan_fn(s, x):
            y, s_new = self.step(x, s, jump_len=jump_len)
            return s_new, y
        
        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        return outputs, final_state
