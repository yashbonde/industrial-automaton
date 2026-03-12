"""PyTorch port of TapeRNN (from models_jax/tape.py).

Faithfully mirrors the JAX architecture:
  - VanillaRNNCell / GRU / LSTM controller
  - Windowed read head (3 cells: pos -1, 0, +1)
  - MLP write head (hidden → 64 → 64 → num_heads * cell_size)
  - Gated write + erase
  - 5 tape actions per head: Stay / Left / Right / JumpLeft / JumpRight
  - Fixed sinusoidal pos_tape (rolled only, never written)
  - Learnable tape initialization
  - PAD freezing via torch.where

Reproducibility:
  All weight tensors are initialized using a seeded torch.Generator
  passed from the trainer. This gives the same initialization for the
  same seed regardless of other global state.

MPS compatibility:
  torch.roll, torch.where, torch.linalg.qr, nn.LSTMCell, nn.GRUCell
  are all supported on the MPS backend (torch 2.10+).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models_torch.common import BaseAutomata


# ── Shared tape-state dataclass ───────────────────────────────────────────────

@dataclass
class TapeRNNState:
    memory:   torch.Tensor  # (num_heads, memory_size, memory_cell_size)
    hidden:   Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    pos_tape: torch.Tensor  # (num_heads, memory_size, pos_dim) — rolled only


# ── Config ────────────────────────────────────────────────────────────────────

class TapeRNNConfig(BaseModel):
    embedding_dim:    int  = 32
    hidden_size:      int  = 256
    memory_size:      int  = 40
    memory_cell_size: int  = 8
    pos_dim:          int  = 16
    use_gru:          bool = False
    use_lstm:         bool = False
    num_heads:        int  = 1


# ── Vanilla RNN cell ──────────────────────────────────────────────────────────

class VanillaRNNCell(nn.Module):
    """h_t = tanh(W [x_t; h_{t-1}] + b)"""
    def __init__(self, input_size: int, hidden_size: int, generator: torch.Generator = None):
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        # Match equinox nn.Linear default: glorot_uniform
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight, generator=generator)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(torch.cat([x, h], dim=-1)))


# ── TapeRNN ───────────────────────────────────────────────────────────────────

class TapeRNN(BaseAutomata):
    """PyTorch TapeRNN — exact feature parity with models_jax/tape.py."""

    def __init__(self, config: TapeRNNConfig, generator: torch.Generator = None):
        super().__init__()

        self.embedding_dim    = config.embedding_dim
        self.hidden_size      = config.hidden_size
        self.memory_size      = config.memory_size
        self.memory_cell_size = config.memory_cell_size
        self.pos_dim          = config.pos_dim
        self.use_gru          = config.use_gru
        self.use_lstm         = config.use_lstm
        self.num_heads        = config.num_heads

        rnn_input_size = config.embedding_dim + config.hidden_size

        # Controller
        if config.use_lstm:
            self.rnn = nn.LSTMCell(rnn_input_size, config.hidden_size)
            self._init_rnn_weights(self.rnn, generator)
        elif config.use_gru:
            self.rnn = nn.GRUCell(rnn_input_size, config.hidden_size)
            self._init_rnn_weights(self.rnn, generator)
        else:
            self.rnn = VanillaRNNCell(rnn_input_size, config.hidden_size, generator=generator)

        scale = 0.1
        read_dim = config.num_heads * (config.memory_cell_size + config.pos_dim) * 3

        # Memory read projection  (hidden_size, read_dim)
        self.W_m = nn.Parameter(torch.randn(config.hidden_size, read_dim, generator=generator) * scale)

        # Action logits  (num_heads*5, hidden_size)
        self.W_a = nn.Parameter(torch.randn(config.num_heads * 5, config.hidden_size, generator=generator) * scale)

        # Action bias — init to strongly favor Right (action index 2)
        b_a = torch.zeros(config.num_heads * 5)
        for h in range(config.num_heads):
            b_a[h * 5 + 2] = 3.0  # Right
        self.b_a = nn.Parameter(b_a)

        # MLP write head
        self.write_l1   = nn.Linear(config.hidden_size, 64)
        self.write_l2   = nn.Linear(64, 64)
        self.write_l3   = nn.Linear(64, config.num_heads * config.memory_cell_size)
        self.write_gate = nn.Linear(config.hidden_size, config.num_heads)
        self.erase_gate = nn.Linear(config.hidden_size, config.num_heads)
        for layer in [self.write_l1, self.write_l2, self.write_l3, self.write_gate, self.erase_gate]:
            with torch.no_grad():
                nn.init.normal_(layer.weight, std=scale, generator=generator)
                nn.init.zeros_(layer.bias)

        # Learnable tape initialization  (memory_size, memory_cell_size)
        self.tape_init = nn.Parameter(
            torch.randn(config.memory_size, config.memory_cell_size, generator=generator) * scale
        )

        # Fixed positional encoding  (memory_size, pos_dim)  — registered as buffer (not learned)
        pe = np.zeros((config.memory_size, config.pos_dim))
        position = np.arange(config.memory_size)[:, np.newaxis]
        div_term = np.exp(np.arange(0, config.pos_dim, 2) * -(np.log(10000.0) / config.pos_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        if config.pos_dim % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        self.register_buffer('pos_tape_init', torch.tensor(pe, dtype=torch.float32))

    @staticmethod
    def _init_rnn_weights(cell, generator):
        with torch.no_grad():
            for name, p in cell.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(p, std=0.1, generator=generator)
                elif 'bias' in name:
                    nn.init.zeros_(p)

    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def init_state(self, device=None) -> TapeRNNState:
        dtype = self.tape_init.dtype
        device = device or self.tape_init.device

        if self.use_lstm:
            hidden = (
                torch.zeros(self.hidden_size, dtype=dtype, device=device),
                torch.zeros(self.hidden_size, dtype=dtype, device=device),
            )
        else:
            hidden = torch.zeros(self.hidden_size, dtype=dtype, device=device)

        return TapeRNNState(
            memory=self.tape_init.unsqueeze(0).expand(self.num_heads, -1, -1).clone(),
            hidden=hidden,
            pos_tape=self.pos_tape_init.unsqueeze(0).expand(self.num_heads, -1, -1).clone(),
        )

    def _step(self, x_t: torch.Tensor, state: TapeRNNState, jump_len) -> Tuple[torch.Tensor, TapeRNNState]:
        """Single timestep (unbatched: x_t is (embedding_dim,)).

        Returns:
            h_new_t: (hidden_size,)
            new_state: updated TapeRNNState
        """
        memory, hidden, pos_tape = state.memory, state.hidden, state.pos_tape

        h_t = hidden[0] if self.use_lstm else hidden  # (hidden_size,)

        # 1. Windowed read — concat [pos-1, pos0, pos+1] for each head
        windows = []
        for h in range(self.num_heads):
            m_h = memory[h]    # (memory_size, memory_cell_size)
            p_h = pos_tape[h]  # (memory_size, pos_dim)
            c_h = torch.cat([m_h, p_h], dim=-1)  # (memory_size, cell+pos_dim)
            w = torch.cat([c_h[-1], c_h[0], c_h[1]], dim=0)
            windows.append(w)

        mem_read = torch.tanh(self.W_m @ torch.cat(windows, dim=0))  # (hidden_size,)

        # 2. RNN step
        rnn_input = torch.cat([x_t, mem_read], dim=0)  # (embedding_dim + hidden_size,)
        if self.use_lstm:
            h_in = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            h_out, c_out = self.rnn(rnn_input.unsqueeze(0), h_in)
            hidden_new = (h_out.squeeze(0), c_out.squeeze(0))
            h_new_t = hidden_new[0]
        elif self.use_gru:
            h_new_t = self.rnn(rnn_input.unsqueeze(0), hidden.unsqueeze(0)).squeeze(0)
            hidden_new = h_new_t
        else:
            h_new_t = self.rnn(rnn_input, h_t)
            hidden_new = h_new_t

        # 3. Tape actions  (num_heads, 5)
        action_logits = self.W_a @ h_new_t + self.b_a  # (num_heads*5,)
        a_t = F.softmax(action_logits.reshape(self.num_heads, 5), dim=-1)

        # 4. Write
        n_t_all = F.relu(self.write_l1(h_new_t))
        n_t_all = F.relu(self.write_l2(n_t_all))
        n_t_all = self.write_l3(n_t_all).reshape(self.num_heads, self.memory_cell_size)

        g_t = torch.sigmoid(self.write_gate(h_new_t))  # (num_heads,)
        e_t = torch.sigmoid(self.erase_gate(h_new_t))  # (num_heads,)

        memory_new_list   = []
        pos_tape_new_list = []

        for h in range(self.num_heads):
            m_h = memory[h]  # (memory_size, memory_cell_size)

            # Gated write at position 0 (differentiable: mask-based, no in-place)
            write_mask = torch.zeros_like(m_h)
            write_mask = write_mask.index_fill(0, torch.zeros(1, dtype=torch.long, device=m_h.device), 1.0)
            written    = (m_h[0] * (1 - e_t[h]) + g_t[h] * n_t_all[h]).unsqueeze(0)  # (1, cell)
            m_h_w = m_h * (1 - write_mask) + written * write_mask

            # Tape roll (differentiable weighted combination of 5 shifts)
            jl = int(jump_len.item()) if hasattr(jump_len, 'item') else int(jump_len)
            m_h_new = (
                a_t[h, 0] * m_h_w +
                a_t[h, 1] * torch.roll(m_h_w, shifts=1,   dims=0) +
                a_t[h, 2] * torch.roll(m_h_w, shifts=-1,  dims=0) +
                a_t[h, 3] * torch.roll(m_h_w, shifts=jl,  dims=0) +
                a_t[h, 4] * torch.roll(m_h_w, shifts=-jl, dims=0)
            )
            memory_new_list.append(m_h_new)

            p_h = pos_tape[h]  # (memory_size, pos_dim)
            p_h_new = (
                a_t[h, 0] * p_h +
                a_t[h, 1] * torch.roll(p_h, shifts=1,   dims=0) +
                a_t[h, 2] * torch.roll(p_h, shifts=-1,  dims=0) +
                a_t[h, 3] * torch.roll(p_h, shifts=jl,  dims=0) +
                a_t[h, 4] * torch.roll(p_h, shifts=-jl, dims=0)
            )
            pos_tape_new_list.append(p_h_new)

        memory_new   = torch.stack(memory_new_list)    # (num_heads, memory_size, cell)
        pos_tape_new = torch.stack(pos_tape_new_list)  # (num_heads, memory_size, pos_dim)

        return h_new_t, hidden_new, memory_new, pos_tape_new

    def forward(self, inputs: torch.Tensor, state: TapeRNNState, pad_mask: torch.Tensor, input_length=None):
        """Process full sequence.

        Args:
            inputs:       (T, embedding_dim)
            state:        TapeRNNState (from init_state)
            pad_mask:     (T,) bool — True for real tokens, False for PAD
            input_length: scalar int tensor — used as jump distance

        Returns:
            hidden_states: (T, hidden_size)
            final_state:   TapeRNNState
        """
        T = inputs.shape[0]
        jump_len = input_length if input_length is not None else torch.tensor(T, device=inputs.device)

        current_state = state
        hidden_states = []

        for t in range(T):
            x_t      = inputs[t]            # (embedding_dim,)
            is_valid = pad_mask[t]           # scalar bool

            h_new_t, hidden_new, memory_new, pos_tape_new = self._step(x_t, current_state, jump_len)

            # Freeze state for PAD tokens — mirrors jnp.where(is_valid, new, old)
            if self.use_lstm:
                frozen_h = tuple(
                    torch.where(is_valid, hn, ho)
                    for hn, ho in zip(hidden_new, current_state.hidden)
                )
            else:
                frozen_h = torch.where(is_valid, hidden_new, current_state.hidden)

            frozen_mem   = torch.where(is_valid.unsqueeze(-1).unsqueeze(-1), memory_new,   current_state.memory)
            frozen_pos   = torch.where(is_valid.unsqueeze(-1).unsqueeze(-1), pos_tape_new, current_state.pos_tape)

            current_state = TapeRNNState(memory=frozen_mem, hidden=frozen_h, pos_tape=frozen_pos)
            hidden_states.append(h_new_t)

        return torch.stack(hidden_states, dim=0), current_state  # (T, hidden_size), state
