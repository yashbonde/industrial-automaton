"""Tallerman architecture: An enhanced TapeRNN with LayerNorm and residual connections."""

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
class TallermanState:
    memory:   torch.Tensor  # (num_heads, memory_size, memory_cell_size)
    hidden:   Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    pos_tape: torch.Tensor  # (num_heads, memory_size, pos_dim) — rolled only


# ── Config ────────────────────────────────────────────────────────────────────

class TallermanConfig(BaseModel):
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
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight, generator=generator)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(torch.cat([x, h], dim=-1)))


# ── Tallerman ───────────────────────────────────────────────────────────────────

class Tallerman(BaseAutomata):
    """Enhanced TapeRNN with LayerNorm and residual connections."""

    def __init__(self, config: TallermanConfig, generator: torch.Generator = None):
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

        # Memory read projection
        self.W_m = nn.Parameter(torch.randn(config.hidden_size, read_dim, generator=generator) * scale)
        self.ln_read = nn.LayerNorm(config.hidden_size)

        # Action logits
        self.W_a = nn.Parameter(torch.randn(config.num_heads * 5, config.hidden_size, generator=generator) * scale)
        b_a = torch.zeros(config.num_heads * 5)
        for h in range(config.num_heads):
            b_a[h * 5 + 2] = 3.0  # Right
        self.b_a = nn.Parameter(b_a)

        # MLP write head
        self.write_l1   = nn.Linear(config.hidden_size, 64)
        self.ln_write1  = nn.LayerNorm(64)
        self.write_l2   = nn.Linear(64, 64)
        self.ln_write2  = nn.LayerNorm(64)
        self.write_l3   = nn.Linear(64, config.num_heads * config.memory_cell_size)
        self.write_gate = nn.Linear(config.hidden_size, config.num_heads)
        self.erase_gate = nn.Linear(config.hidden_size, config.num_heads)
        
        for layer in [self.write_l1, self.write_l2, self.write_l3, self.write_gate, self.erase_gate]:
            with torch.no_grad():
                nn.init.normal_(layer.weight, std=scale, generator=generator)
                nn.init.zeros_(layer.bias)

        # Learnable tape initialization
        self.tape_init = nn.Parameter(
            torch.randn(config.memory_size, config.memory_cell_size, generator=generator) * scale
        )

        # Fixed positional encoding
        pe = np.zeros((config.memory_size, config.pos_dim))
        position = np.arange(config.memory_size)[:, np.newaxis]
        div_term = np.exp(np.arange(0, config.pos_dim, 2) * -(np.log(10000.0) / config.pos_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        if config.pos_dim % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        self.register_buffer('pos_tape_init', torch.tensor(pe, dtype=torch.float32))

        # Output projection and normalization
        self.ln_out = nn.LayerNorm(config.hidden_size)

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

    def init_state(self, device=None) -> TallermanState:
        dtype = self.tape_init.dtype
        device = device or self.tape_init.device

        if self.use_lstm:
            hidden = (
                torch.zeros(self.hidden_size, dtype=dtype, device=device),
                torch.zeros(self.hidden_size, dtype=dtype, device=device),
            )
        else:
            hidden = torch.zeros(self.hidden_size, dtype=dtype, device=device)

        return TallermanState(
            memory=self.tape_init.unsqueeze(0).expand(self.num_heads, -1, -1).clone(),
            hidden=hidden,
            pos_tape=self.pos_tape_init.unsqueeze(0).expand(self.num_heads, -1, -1).clone(),
        )

    def _step(self, x_t: torch.Tensor, state: TallermanState, jump_len) -> Tuple[torch.Tensor, TallermanState]:
        memory, hidden, pos_tape = state.memory, state.hidden, state.pos_tape
        h_t = hidden[0] if self.use_lstm else hidden

        # 1. Windowed read
        windows = []
        for h in range(self.num_heads):
            m_h = memory[h]
            p_h = pos_tape[h]
            c_h = torch.cat([m_h, p_h], dim=-1)
            # Roll for windowed read
            w = torch.cat([torch.roll(c_h, 1, 0)[0], c_h[0], torch.roll(c_h, -1, 0)[0]], dim=0)
            windows.append(w)

        mem_read = self.ln_read(torch.tanh(self.W_m @ torch.cat(windows, dim=0)))

        # 2. RNN step
        rnn_input = torch.cat([x_t, mem_read], dim=0)
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

        # LayerNorm and Residual for hidden state
        h_new_t = self.ln_out(h_new_t + h_t)

        # 3. Tape actions
        action_logits = self.W_a @ h_new_t + self.b_a
        a_t = F.softmax(action_logits.reshape(self.num_heads, 5), dim=-1)

        # 4. Write
        n_t_all = F.relu(self.ln_write1(self.write_l1(h_new_t)))
        n_t_all = F.relu(self.ln_write2(self.write_l2(n_t_all)))
        n_t_all = self.write_l3(n_t_all).reshape(self.num_heads, self.memory_cell_size)

        g_t = torch.sigmoid(self.write_gate(h_new_t))
        e_t = torch.sigmoid(self.erase_gate(h_new_t))

        memory_new_list   = []
        pos_tape_new_list = []

        for h in range(self.num_heads):
            m_h = memory[h]
            # Gated write at position 0
            write_mask = torch.zeros(self.memory_size, 1, device=m_h.device)
            write_mask[0, 0] = 1.0
            
            written = m_h[0] * (1 - e_t[h]) + g_t[h] * n_t_all[h]
            m_h_w = m_h * (1 - write_mask) + written.unsqueeze(0) * write_mask

            # Tape roll
            jl = int(jump_len.item()) if hasattr(jump_len, 'item') else int(jump_len)
            m_h_new = (
                a_t[h, 0] * m_h_w +
                a_t[h, 1] * torch.roll(m_h_w, shifts=1,   dims=0) +
                a_t[h, 2] * torch.roll(m_h_w, shifts=-1,  dims=0) +
                a_t[h, 3] * torch.roll(m_h_w, shifts=jl,  dims=0) +
                a_t[h, 4] * torch.roll(m_h_w, shifts=-jl, dims=0)
            )
            memory_new_list.append(m_h_new)

            p_h = pos_tape[h]
            p_h_new = (
                a_t[h, 0] * p_h +
                a_t[h, 1] * torch.roll(p_h, shifts=1,   dims=0) +
                a_t[h, 2] * torch.roll(p_h, shifts=-1,  dims=0) +
                a_t[h, 3] * torch.roll(p_h, shifts=jl,  dims=0) +
                a_t[h, 4] * torch.roll(p_h, shifts=-jl, dims=0)
            )
            pos_tape_new_list.append(p_h_new)

        memory_new   = torch.stack(memory_new_list)
        pos_tape_new = torch.stack(pos_tape_new_list)

        return h_new_t, hidden_new, memory_new, pos_tape_new

    def forward(self, inputs: torch.Tensor, state: TallermanState, pad_mask: torch.Tensor, input_length=None):
        T = inputs.shape[0]
        jump_len = input_length if input_length is not None else torch.tensor(T, device=inputs.device)

        current_state = state
        hidden_states = []

        for t in range(T):
            x_t      = inputs[t]
            is_valid = pad_mask[t]

            h_new_t, hidden_new, memory_new, pos_tape_new = self._step(x_t, current_state, jump_len)

            if self.use_lstm:
                frozen_h = tuple(
                    torch.where(is_valid, hn, ho)
                    for hn, ho in zip(hidden_new, current_state.hidden)
                )
            else:
                frozen_h = torch.where(is_valid, hidden_new, current_state.hidden)

            frozen_mem   = torch.where(is_valid.unsqueeze(-1).unsqueeze(-1), memory_new,   current_state.memory)
            frozen_pos   = torch.where(is_valid.unsqueeze(-1).unsqueeze(-1), pos_tape_new, current_state.pos_tape)

            current_state = TallermanState(memory=frozen_mem, hidden=frozen_h, pos_tape=frozen_pos)
            hidden_states.append(h_new_t)

        return torch.stack(hidden_states, dim=0), current_state
