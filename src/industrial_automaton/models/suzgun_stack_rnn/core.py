"""Suzgun Stack-RNN (Suzgun et al. 2019)."""

from typing import NamedTuple
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import equinox as eqx

from industrial_automaton.vocab import SIZE as VOCAB_SIZE

class StackRNNState(NamedTuple):
    stack: jnp.ndarray    # (Depth, Dim)
    hidden: jnp.ndarray   # (Hidden,)

class SuzgunStackRNNConfig(BaseModel):
    hidden_size: int = 16
    stack_depth: int = 50
    value_dim: int = 1

class SuzgunStackRNN(eqx.Module):
    """Suzgun Stack-RNN: RNN controller + differentiable stack.
    
    Architecture (Suzgun et al. 2019):
    h_tilde_{t-1} = h_{t-1} + W_sh * stack_{t-1}[0]
    h_t = tanh(W_ih * x_t + b_ih + W_hh * h_tilde_{t-1} + b_hh)
    y_t = W_y * h_t
    a_t = softmax(W_a * h_t)  # [push, pop, no-op]
    n_t = sigmoid(W_n * h_t)  # value to push
    """
    
    W_ih: jnp.ndarray
    b_ih: jnp.ndarray
    W_hh: jnp.ndarray
    b_hh: jnp.ndarray
    W_sh: jnp.ndarray
    W_y: jnp.ndarray
    W_a: jnp.ndarray
    W_n: jnp.ndarray
    
    hidden_size: int = eqx.field(static=True)
    stack_depth: int = eqx.field(static=True)
    value_dim: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    def __init__(
        self,
        config: SuzgunStackRNNConfig,
        *,
        key,
    ):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.vocab_size = VOCAB_SIZE
        self.hidden_size = config.hidden_size
        self.stack_depth = config.stack_depth
        self.value_dim = config.value_dim

        scale = 1.0 / (config.hidden_size**0.5)
        self.W_ih = jax.random.uniform(k1, (config.hidden_size, VOCAB_SIZE), minval=-scale, maxval=scale)
        self.b_ih = jnp.zeros(config.hidden_size)
        self.W_hh = jax.random.uniform(k2, (config.hidden_size, config.hidden_size), minval=-scale, maxval=scale)
        self.b_hh = jnp.zeros(config.hidden_size)

        self.W_sh = jax.random.normal(k3, (config.hidden_size, config.value_dim)) * 0.1
        self.W_y = jax.random.normal(k4, (VOCAB_SIZE, config.hidden_size)) * 0.1
        self.W_a = jax.random.normal(k5, (3, config.hidden_size)) * 0.1
        self.W_n = jax.random.normal(k6, (config.value_dim, config.hidden_size)) * 0.1

    def init_state(self) -> StackRNNState:
        return StackRNNState(
            stack=jnp.zeros((self.stack_depth, self.value_dim)),
            hidden=jnp.zeros(self.hidden_size),
        )

    def step(self, x_t: jnp.ndarray, state: StackRNNState):
        stack, hidden = state
        
        # 1. Inject stack top into hidden state (Suzgun style: additive)
        h_tilde = hidden + self.W_sh @ stack[0]
        
        # 2. Update hidden state (RNN)
        hidden_new = jnp.tanh(self.W_ih @ x_t + self.b_ih + self.W_hh @ h_tilde + self.b_hh)
        
        # 3. Output
        y_t = self.W_y @ hidden_new
        
        # 4. Action weights (PUSH, POP, NO-OP)
        a_t = jax.nn.softmax(self.W_a @ hidden_new)
        push_weight = a_t[0]
        pop_weight = a_t[1]
        noop_weight = a_t[2]
        
        # 5. Value to push
        n_t = jax.nn.sigmoid(self.W_n @ hidden_new)
        
        # 6. Update stack (differentiable)
        # s_t[0] = push * n_t + pop * stack[1] + noop * stack[0]
        # s_t[i] = push * stack[i-1] + pop * stack[i+1] + noop * stack[i]
        
        stack_push = jnp.concatenate([n_t[None], stack[:-1]], axis=0)
        stack_pop = jnp.concatenate([stack[1:], jnp.zeros((1, self.value_dim))], axis=0)
        stack_noop = stack
        
        stack_new = push_weight * stack_push + pop_weight * stack_pop + noop_weight * stack_noop
        
        return y_t, StackRNNState(stack_new, hidden_new)

    def __call__(self, inputs: jnp.ndarray, state: StackRNNState):
        """Process sequence. inputs: (T, vocab) one-hot. Returns (outputs (T, vocab), final_state)."""
        def scan_fn(state, x_t):
            y_t, new_state = self.step(x_t, state)
            return new_state, y_t
        
        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        return outputs, final_state
