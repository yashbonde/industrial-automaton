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
    use_pos_attn:     bool = True
    write_hidden:     int  = 64
    window_size:      int  = 3
    use_input_write:  bool = False  # Add input embedding residual to write vector


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
        self.use_pos_attn     = config.use_pos_attn
        self.window_size      = config.window_size

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
        actual_window_slots = 2 * (config.window_size // 2) + 1
        read_dim = config.num_heads * (config.memory_cell_size + config.pos_dim) * actual_window_slots

        # Memory read projection (windowed positional)
        self.W_m = nn.Parameter(torch.randn(config.hidden_size, read_dim, generator=generator) * scale)
        self.ln_read = nn.LayerNorm(config.hidden_size)

        # Content-based attention read
        self.W_q = nn.Parameter(torch.randn(config.memory_cell_size, config.hidden_size, generator=generator) * scale)
        self.W_cv = nn.Parameter(torch.randn(config.hidden_size, config.num_heads * config.memory_cell_size, generator=generator) * scale)
        self.ln_content = nn.LayerNorm(config.hidden_size)
        self._cell_scale = math.sqrt(config.memory_cell_size)

        # Positional attention read: query pos_tape to locate target position, read memory there
        if config.use_pos_attn:
            self.W_pq = nn.Parameter(torch.randn(config.pos_dim, config.hidden_size, generator=generator) * scale)
            self.W_pv = nn.Parameter(torch.randn(config.hidden_size, config.num_heads * config.memory_cell_size, generator=generator) * scale)
            self.ln_pos_read = nn.LayerNorm(config.hidden_size)
            self._pos_scale = math.sqrt(config.pos_dim)

        # Action logits
        self.W_a = nn.Parameter(torch.randn(config.num_heads * 5, config.hidden_size, generator=generator) * scale)
        b_a = torch.zeros(config.num_heads * 5)
        for h in range(config.num_heads):
            b_a[h * 5 + 2] = 3.0  # Right
        self.b_a = nn.Parameter(b_a)

        # MLP write head
        write_hidden = config.write_hidden
        self.write_l1   = nn.Linear(config.hidden_size, write_hidden)
        self.ln_write1  = nn.LayerNorm(write_hidden)
        self.write_l2   = nn.Linear(write_hidden, write_hidden)
        self.ln_write2  = nn.LayerNorm(write_hidden)
        self.write_l3   = nn.Linear(write_hidden, config.num_heads * config.memory_cell_size)
        self.write_gate = nn.Linear(config.hidden_size, config.num_heads)
        self.erase_gate = nn.Linear(config.hidden_size, config.num_heads)
        # Direct input residual for write: stores raw token embedding alongside MLP output
        self.use_input_write = config.use_input_write
        if config.use_input_write:
            self.W_xi = nn.Parameter(torch.randn(config.memory_cell_size, config.embedding_dim, generator=generator) * scale)
        
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

    def init_state(self, batch_size=1, device=None) -> TallermanState:
        dtype = self.tape_init.dtype
        device = device or self.tape_init.device

        if self.use_lstm:
            hidden = (
                torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device),
                torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device),
            )
        else:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)

        return TallermanState(
            memory=self.tape_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1).clone(),
            hidden=hidden,
            pos_tape=self.pos_tape_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1).clone(),
        )

    def _step(self, x_t: torch.Tensor, state: TallermanState, jump_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t:      (B, embedding_dim)
            state:    TallermanState with:
                      memory:   (B, num_heads, memory_size, memory_cell_size)
                      hidden:   (B, hidden_size) or ((B, hidden_size), (B, hidden_size))
                      pos_tape: (B, num_heads, memory_size, pos_dim)
            jump_len: (B,) int tensor for shifts=jl, shifts=-jl

        Returns:
            h_new_t:    (B, hidden_size)
            hidden_new: (B, hidden_size) or tuple (LSTM state)
            memory_new: (B, num_heads, memory_size, memory_cell_size)
            pos_tape_new: (B, num_heads, memory_size, pos_dim)
        """
        memory, hidden, pos_tape = state.memory, state.hidden, state.pos_tape
        h_t = hidden[0] if self.use_lstm else hidden
        B = x_t.shape[0]

        # 1. Windowed read
        # Concatenate memory and positional tape for reading
        combined = torch.cat([memory, pos_tape], dim=-1)  # (B, num_heads, memory_size, cell+pos)
        
        # Windowed read: read window_size slots centered at position 0
        half = self.window_size // 2
        slots = [torch.roll(combined, shifts=s, dims=2)[:, :, 0]
                 for s in range(-half, half + 1)]
        window = torch.cat(slots, dim=-1).reshape(B, -1)
        
        mem_read = self.ln_read(torch.tanh(window @ self.W_m.T))

        # Content-based attention read
        query = h_t @ self.W_q.T  # (B, memory_cell_size)
        # scores: (B, num_heads, memory_size)
        scores = torch.einsum('bc,bnmc->bnm', query, memory) / self._cell_scale
        attn = F.softmax(scores, dim=-1)  # (B, num_heads, memory_size)
        # weighted sum of memory cells: (B, num_heads, memory_cell_size)
        content_vec = torch.einsum('bnm,bnmc->bnc', attn, memory)
        content_read = self.ln_content(content_vec.reshape(B, -1) @ self.W_cv.T)

        # Positional attention: query pos_tape to locate target position, read memory there
        if self.use_pos_attn:
            pos_query = h_t @ self.W_pq.T  # (B, pos_dim)
            pos_scores = torch.einsum('bc,bnmc->bnm', pos_query, pos_tape) / self._pos_scale
            pos_attn = F.softmax(pos_scores, dim=-1)  # (B, num_heads, memory_size)
            pos_value = torch.einsum('bnm,bnmc->bnc', pos_attn, memory)  # read memory at attended pos
            pos_read = self.ln_pos_read(pos_value.reshape(B, -1) @ self.W_pv.T)
        else:
            pos_read = 0

        # 2. RNN step
        rnn_input = torch.cat([x_t, mem_read + content_read + pos_read], dim=-1)
        if self.use_lstm:
            hidden_new = self.rnn(rnn_input, hidden)
            h_new_t = hidden_new[0]
        elif self.use_gru:
            hidden_new = self.rnn(rnn_input, hidden)
            h_new_t = hidden_new
        else:
            h_new_t = self.rnn(rnn_input, h_t)
            hidden_new = h_new_t

        # LayerNorm and Residual for hidden state
        h_new_t = self.ln_out(h_new_t + h_t)

        # 3. Tape actions
        action_logits = h_new_t @ self.W_a.T + self.b_a
        a_t = F.softmax(action_logits.reshape(B, self.num_heads, 5), dim=-1)

        # 4. Write
        n_t_all = F.gelu(self.ln_write1(self.write_l1(h_new_t)))
        n_t_all = F.gelu(self.ln_write2(self.write_l2(n_t_all)))
        n_t_all = self.write_l3(n_t_all).reshape(B, self.num_heads, self.memory_cell_size)
        if self.use_input_write:
            n_t_all = n_t_all + (x_t @ self.W_xi.T).unsqueeze(1)

        g_t = torch.sigmoid(self.write_gate(h_new_t)).unsqueeze(-1)  # (B, num_heads, 1)
        e_t = torch.sigmoid(self.erase_gate(h_new_t)).unsqueeze(-1)  # (B, num_heads, 1)

        # Gated write at position 0
        # memory: (B, num_heads, memory_size, cell_size)
        written = memory[:, :, 0] * (1 - e_t) + g_t * n_t_all # (B, num_heads, cell_size)
        
        m_h_w = memory.clone()
        m_h_w[:, :, 0] = written

        # Tape roll - vectorizing across batch for fixed shifts
        # shifts = 0, 1, -1, jl, -jl
        # We need to handle per-batch jump lengths. 
        # For simplicity, if jl is uniform across batch (usual case in current trainer), 
        # we can still use torch.roll.
        
        jl = int(jump_len[0].item()) if jump_len.dim() > 0 else int(jump_len.item())
        
        def roll_and_weight(tensor, action_weights):
            # tensor: (B, num_heads, memory_size, D)
            # action_weights: (B, num_heads, 5)
            r0 = tensor
            r1 = torch.roll(tensor, shifts=1,   dims=2)
            r2 = torch.roll(tensor, shifts=-1,  dims=2)
            r3 = torch.roll(tensor, shifts=jl,  dims=2)
            r4 = torch.roll(tensor, shifts=-jl, dims=2)
            
            w = action_weights.unsqueeze(-1).unsqueeze(-1) # (B, num_heads, 5, 1, 1)
            stacked = torch.stack([r0, r1, r2, r3, r4], dim=2) # (B, num_heads, 5, memory_size, D)
            return (w * stacked).sum(dim=2)

        memory_new = roll_and_weight(m_h_w, a_t)
        pos_tape_new = roll_and_weight(pos_tape, a_t)

        return h_new_t, hidden_new, memory_new, pos_tape_new

    def forward(self, inputs: torch.Tensor, state: TallermanState, pad_mask: torch.Tensor, input_length=None):
        """
        Args:
            inputs: (B, T, embedding_dim)
            state: TallermanState
            pad_mask: (B, T) bool
            input_length: (B,) int
        """
        B, T, _ = inputs.shape
        if input_length is None:
            input_length = torch.full((B,), T, device=inputs.device, dtype=torch.long)
        elif not isinstance(input_length, torch.Tensor):
            input_length = torch.full((B,), input_length, device=inputs.device, dtype=torch.long)
        
        # jump_len must be uniform across batch for vectorized roll to be simple.
        # Current trainer uses uniform length per batch.
        jump_len = input_length

        current_state = state
        hidden_states = []

        for t in range(T):
            x_t      = inputs[:, t]        # (B, embedding_dim)
            is_valid = pad_mask[:, t]     # (B,)

            h_new_t, hidden_new, memory_new, pos_tape_new = self._step(x_t, current_state, jump_len)

            # Update state only for valid tokens
            if self.use_lstm:
                new_h = (
                    torch.where(is_valid.unsqueeze(-1), hidden_new[0], current_state.hidden[0]),
                    torch.where(is_valid.unsqueeze(-1), hidden_new[1], current_state.hidden[1])
                )
            else:
                new_h = torch.where(is_valid.unsqueeze(-1), hidden_new, current_state.hidden)

            new_mem = torch.where(is_valid.view(B, 1, 1, 1), memory_new, current_state.memory)
            new_pos = torch.where(is_valid.view(B, 1, 1, 1), pos_tape_new, current_state.pos_tape)

            current_state = TallermanState(memory=new_mem, hidden=new_h, pos_tape=new_pos)
            hidden_states.append(h_new_t)

        return torch.stack(hidden_states, dim=1), current_state
