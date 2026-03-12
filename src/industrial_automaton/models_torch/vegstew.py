"""Vegstew architecture: Taking a DNC and applying learning from Tallerman experiments (mar11)

VegStew = Tallerman + DNC write improvements + K/V memory split.
Key differences from Tallerman:
  - Memory cell split into key_size + val_size halves
  - Content attention queries key-half, reads value-half
  - DNC-style soft write: content_w*(1-alloc_gate) + alloc_w*alloc_gate
  - Gaussian locality prior modulating write weights (sigma learned per head)
  - Per-word erase vector (DNC) instead of scalar erase gate
  - EMA-based usage tracking for allocation
  - Learned beta (sharpening) per head for content attention
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton.models_torch.common import BaseAutomata


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class VegStewState:
    memory:    torch.Tensor  # (B, num_heads, N, key_size + val_size)
    hidden:    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    pos_tape:  torch.Tensor  # (B, num_heads, N, pos_dim)
    usage:     torch.Tensor  # (B, N) EMA-based usage
    write_w:   torch.Tensor  # (B, num_heads, N) last write weights (for usage update)


# ── Config ────────────────────────────────────────────────────────────────────

class VegStewConfig(BaseModel):
    embedding_dim: int  = 16
    hidden_size:   int  = 256
    memory_size:   int  = 60
    key_size:      int  = 8    # key half of memory cell
    val_size:      int  = 8    # value half of memory cell
    pos_dim:       int  = 16
    use_gru:       bool = False
    use_lstm:      bool = False
    num_heads:     int  = 1
    use_pos_attn:  bool = True
    write_hidden:  int  = 64
    window_size:   int  = 3
    use_input_write: bool = False
    ema_decay:     float = 0.99   # EMA decay for usage tracking
    use_soft_write: bool = True   # DNC-style soft write; False = Tallerman slot-0 write (K/V split kept)


# ── Vanilla RNN cell ──────────────────────────────────────────────────────────

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, generator=None):
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight, generator=generator)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, h):
        return torch.tanh(self.linear(torch.cat([x, h], dim=-1)))


# ── VegStew ───────────────────────────────────────────────────────────────────

class VegStew(BaseAutomata):
    """VegStew: Tallerman + DNC write + K/V memory split."""

    def __init__(self, config: VegStewConfig, generator=None):
        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.hidden_size   = config.hidden_size
        self.memory_size   = config.memory_size
        self.key_size      = config.key_size
        self.val_size      = config.val_size
        self.cell_size     = config.key_size + config.val_size
        self.pos_dim       = config.pos_dim
        self.use_gru       = config.use_gru
        self.use_lstm      = config.use_lstm
        self.num_heads     = config.num_heads
        self.use_pos_attn  = config.use_pos_attn
        self.window_size    = config.window_size
        self.ema_decay      = config.ema_decay
        self.use_soft_write = config.use_soft_write

        scale = 0.1
        actual_window_slots = 2 * (config.window_size // 2) + 1
        # Windowed read uses full cell + pos_tape (same as Tallerman)
        read_dim = config.num_heads * (self.cell_size + config.pos_dim) * actual_window_slots

        # Controller
        rnn_input_size = config.embedding_dim + config.hidden_size
        if config.use_lstm:
            self.rnn = nn.LSTMCell(rnn_input_size, config.hidden_size)
            self._init_rnn_weights(self.rnn, generator)
        elif config.use_gru:
            self.rnn = nn.GRUCell(rnn_input_size, config.hidden_size)
            self._init_rnn_weights(self.rnn, generator)
        else:
            self.rnn = VanillaRNNCell(rnn_input_size, config.hidden_size, generator=generator)

        # Windowed read projection
        self.W_m = nn.Parameter(torch.randn(config.hidden_size, read_dim, generator=generator) * scale)
        self.ln_read = nn.LayerNorm(config.hidden_size)

        # Content read: query key-half, read val-half
        # W_q: h_t → key_size query
        self.W_q = nn.Parameter(torch.randn(config.key_size, config.hidden_size, generator=generator) * scale)
        # W_cv: val-half weighted sum → hidden_size
        self.W_cv = nn.Parameter(torch.randn(config.hidden_size, config.num_heads * config.val_size, generator=generator) * scale)
        self.ln_content = nn.LayerNorm(config.hidden_size)
        # Learned sharpening per head (β)
        self.log_beta = nn.Parameter(torch.zeros(config.num_heads))  # β = exp(log_beta)

        # Positional attention: query pos_tape, read val-half
        if config.use_pos_attn:
            self.W_pq = nn.Parameter(torch.randn(config.pos_dim, config.hidden_size, generator=generator) * scale)
            self.W_pv = nn.Parameter(torch.randn(config.hidden_size, config.num_heads * config.val_size, generator=generator) * scale)
            self.ln_pos_read = nn.LayerNorm(config.hidden_size)
            self._pos_scale = math.sqrt(config.pos_dim)

        # Action logits (5 actions: stay/left/right/jump+/jump-)
        self.W_a = nn.Parameter(torch.randn(config.num_heads * 5, config.hidden_size, generator=generator) * scale)
        b_a = torch.zeros(config.num_heads * 5)
        for h in range(config.num_heads):
            b_a[h * 5 + 2] = 3.0  # bias toward "right"
        self.b_a = nn.Parameter(b_a)

        # MLP write head (writes to full cell_size)
        wh = config.write_hidden
        self.write_l1   = nn.Linear(config.hidden_size, wh)
        self.ln_write1  = nn.LayerNorm(wh)
        self.write_l2   = nn.Linear(wh, wh)
        self.ln_write2  = nn.LayerNorm(wh)
        self.write_l3   = nn.Linear(wh, config.num_heads * self.cell_size)
        for layer in [self.write_l1, self.write_l2, self.write_l3]:
            with torch.no_grad():
                nn.init.normal_(layer.weight, std=scale, generator=generator)
                nn.init.zeros_(layer.bias)

        self.use_input_write = config.use_input_write
        if config.use_input_write:
            self.W_xi = nn.Parameter(torch.randn(self.cell_size, config.embedding_dim, generator=generator) * scale)

        # Write gate (scalar per head) — used in both write paths
        self.write_gate = nn.Linear(config.hidden_size, config.num_heads)
        with torch.no_grad():
            nn.init.normal_(self.write_gate.weight, std=scale, generator=generator)
            nn.init.zeros_(self.write_gate.bias)

        # Per-word erase vector (DNC soft write path): sigmoid linear → (num_heads * cell_size,)
        self.W_erase = nn.Linear(config.hidden_size, config.num_heads * self.cell_size)
        with torch.no_grad():
            nn.init.normal_(self.W_erase.weight, std=scale, generator=generator)
            nn.init.zeros_(self.W_erase.bias)

        # Scalar erase gate (simple write path, Tallerman-style at slot 0)
        self.erase_gate = nn.Linear(config.hidden_size, config.num_heads)
        with torch.no_grad():
            nn.init.normal_(self.erase_gate.weight, std=scale, generator=generator)
            nn.init.zeros_(self.erase_gate.bias)

        if config.use_soft_write:
            # Allocation gate (interpolates content-write vs alloc-write)
            self.W_alloc_gate = nn.Linear(config.hidden_size, config.num_heads)
            with torch.no_grad():
                nn.init.normal_(self.W_alloc_gate.weight, std=scale, generator=generator)
                nn.init.zeros_(self.W_alloc_gate.bias)

            # Write content key (to compute content-based write address)
            self.W_wq = nn.Parameter(torch.randn(config.key_size, config.hidden_size, generator=generator) * scale)

            # Gaussian locality prior: learned log_sigma per head (σ = exp(log_sigma))
            self.log_sigma = nn.Parameter(torch.ones(config.num_heads) * math.log(4.0))

        # Learnable tape init
        self.tape_init = nn.Parameter(
            torch.randn(config.memory_size, self.cell_size, generator=generator) * scale
        )

        # Fixed sinusoidal positional tape
        pe = np.zeros((config.memory_size, config.pos_dim))
        position = np.arange(config.memory_size)[:, np.newaxis]
        div_term = np.exp(np.arange(0, config.pos_dim, 2) * -(math.log(10000.0) / config.pos_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        if config.pos_dim % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        self.register_buffer('pos_tape_init', torch.tensor(pe, dtype=torch.float32))

        if config.use_soft_write:
            # Slot distance buffer for locality prior
            d = torch.arange(config.memory_size, dtype=torch.float32)
            d = torch.min(d, config.memory_size - d)
            self.register_buffer('slot_dist', d)  # (N,)

        # Output norm
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

    def init_state(self, batch_size=1, device=None) -> VegStewState:
        dtype = self.tape_init.dtype
        device = device or self.tape_init.device

        if self.use_lstm:
            hidden = (
                torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device),
                torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device),
            )
        else:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)

        return VegStewState(
            memory=self.tape_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1).clone(),
            hidden=hidden,
            pos_tape=self.pos_tape_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1).clone(),
            usage=torch.zeros(batch_size, self.memory_size, dtype=dtype, device=device),
            write_w=torch.zeros(batch_size, self.num_heads, self.memory_size, dtype=dtype, device=device),
        )

    def _step(self, x_t: torch.Tensor, state: VegStewState, jump_len: torch.Tensor):
        memory, hidden, pos_tape = state.memory, state.hidden, state.pos_tape
        usage, prev_write_w = state.usage, state.write_w
        h_t = hidden[0] if self.use_lstm else hidden
        B = x_t.shape[0]
        N = self.memory_size
        H = self.num_heads

        key_half = memory[..., :self.key_size]   # (B, H, N, key_size)
        val_half = memory[..., self.key_size:]    # (B, H, N, val_size)

        # ── 1. Windowed read (full cell + pos_tape, same as Tallerman) ──────
        combined = torch.cat([memory, pos_tape], dim=-1)  # (B, H, N, cell+pos)
        half = self.window_size // 2
        slots = [torch.roll(combined, shifts=s, dims=2)[:, :, 0]
                 for s in range(-half, half + 1)]
        window = torch.cat(slots, dim=-1).reshape(B, -1)
        mem_read = self.ln_read(torch.tanh(window @ self.W_m.T))  # (B, hidden_size)

        # ── 2. Content read: query key-half, read val-half ──────────────────
        beta = self.log_beta.exp()  # (H,)
        query = h_t @ self.W_q.T   # (B, key_size)
        # scores: (B, H, N)
        scores = torch.einsum('bk,bhnk->bhn', query, key_half) * beta.view(1, H, 1) / math.sqrt(self.key_size)
        attn = F.softmax(scores, dim=-1)  # (B, H, N)
        content_vec = torch.einsum('bhn,bhnv->bhv', attn, val_half)  # (B, H, val_size)
        content_read = self.ln_content(content_vec.reshape(B, -1) @ self.W_cv.T)  # (B, hidden_size)

        # ── 3. Positional attention: query pos_tape, read val-half ──────────
        if self.use_pos_attn:
            pos_query = h_t @ self.W_pq.T  # (B, pos_dim)
            pos_scores = torch.einsum('bp,bhnp->bhn', pos_query, pos_tape) / self._pos_scale
            pos_attn = F.softmax(pos_scores, dim=-1)  # (B, H, N)
            pos_vec = torch.einsum('bhn,bhnv->bhv', pos_attn, val_half)  # (B, H, val_size)
            pos_read = self.ln_pos_read(pos_vec.reshape(B, -1) @ self.W_pv.T)  # (B, hidden_size)
        else:
            pos_read = 0

        # ── 4. Controller ────────────────────────────────────────────────────
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

        # ── 5. LN + residual ─────────────────────────────────────────────────
        h_new_t = self.ln_out(h_new_t + h_t)

        # ── 6. Write value (MLP) ─────────────────────────────────────────────
        n_t = F.gelu(self.ln_write1(self.write_l1(h_new_t)))
        n_t = F.gelu(self.ln_write2(self.write_l2(n_t)))
        n_t = self.write_l3(n_t).reshape(B, H, self.cell_size)  # (B, H, cell_size)
        if self.use_input_write:
            n_t = n_t + (x_t @ self.W_xi.T).unsqueeze(1)

        if self.use_soft_write:
            # ── 7. Per-word erase vector (DNC) ───────────────────────────────
            e_t = torch.sigmoid(self.W_erase(h_new_t)).reshape(B, H, self.cell_size)

            # ── 8. Write address ─────────────────────────────────────────────
            write_query = h_new_t @ self.W_wq.T  # (B, key_size)
            w_scores = torch.einsum('bk,bhnk->bhn', write_query, key_half) / math.sqrt(self.key_size)
            content_w = F.softmax(w_scores, dim=-1)

            alloc_w = F.softmax(-usage.unsqueeze(1).expand(B, H, N), dim=-1)
            alloc_gate = torch.sigmoid(self.W_alloc_gate(h_new_t)).unsqueeze(-1)
            write_addr = (1 - alloc_gate) * content_w + alloc_gate * alloc_w

            sigma = self.log_sigma.exp().clamp(min=0.5)
            d2 = self.slot_dist.pow(2)
            locality = torch.exp(-d2.view(1, 1, N) / (2 * sigma.view(1, H, 1).pow(2)))
            write_addr = write_addr * locality
            write_addr = write_addr / (write_addr.sum(dim=-1, keepdim=True) + 1e-8)

            g_t = torch.sigmoid(self.write_gate(h_new_t)).unsqueeze(-1)
            w_final = g_t * write_addr

            # ── 9. Update usage (EMA) ─────────────────────────────────────────
            write_w_sum = w_final.sum(dim=1)
            usage_new = self.ema_decay * usage + (1 - self.ema_decay) * write_w_sum

            # ── 10. Write: M *= (1 - w⊗e); M += w⊗v ─────────────────────────
            erase = w_final.unsqueeze(-1) * e_t.unsqueeze(2)
            add   = w_final.unsqueeze(-1) * n_t.unsqueeze(2)
            memory_new = memory * (1 - erase) + add
        else:
            # ── Simple write: Tallerman slot-0, per-word erase, K/V split ────
            g_t  = torch.sigmoid(self.write_gate(h_new_t)).unsqueeze(-1)   # (B, H, 1)
            e_t  = torch.sigmoid(self.erase_gate(h_new_t)).unsqueeze(-1)   # (B, H, 1) scalar per head
            # Write to slot 0 only
            written = memory[:, :, 0] * (1 - e_t) + g_t * n_t             # (B, H, cell_size)
            memory_new = memory.clone()
            memory_new[:, :, 0] = written
            usage_new = usage   # unchanged — not tracked in simple write
            w_final = None      # not used in simple write

        # ── 11. Tape actions ─────────────────────────────────────────────────
        action_logits = h_new_t @ self.W_a.T + self.b_a
        a_t = F.softmax(action_logits.reshape(B, H, 5), dim=-1)

        # ── 12. Roll memory + pos_tape ───────────────────────────────────────
        jl = int(jump_len[0].item()) if jump_len.dim() > 0 else int(jump_len.item())

        def roll_and_weight(tensor, action_weights):
            r0 = tensor
            r1 = torch.roll(tensor, shifts=1,   dims=2)
            r2 = torch.roll(tensor, shifts=-1,  dims=2)
            r3 = torch.roll(tensor, shifts=jl,  dims=2)
            r4 = torch.roll(tensor, shifts=-jl, dims=2)
            w = action_weights.unsqueeze(-1).unsqueeze(-1)
            stacked = torch.stack([r0, r1, r2, r3, r4], dim=2)
            return (w * stacked).sum(dim=2)

        memory_rolled = roll_and_weight(memory_new, a_t)
        pos_tape_rolled = roll_and_weight(pos_tape, a_t)

        # Also roll usage and write_w consistently (slot 0 is current position)
        # We track write_w for the next step's usage (already updated above)
        # Roll write_w_sum for usage alignment (usage is per-slot global)
        # Actually usage doesn't need to be rolled — it's a global slot allocation tracker
        # We keep it unrolled so allocation is globally aware of which slots are used.

        return h_new_t, hidden_new, memory_rolled, pos_tape_rolled, usage_new, w_final

    def forward(self, inputs: torch.Tensor, state: VegStewState, pad_mask: torch.Tensor, input_length=None):
        B, T, _ = inputs.shape
        if input_length is None:
            input_length = torch.full((B,), T, device=inputs.device, dtype=torch.long)
        elif not isinstance(input_length, torch.Tensor):
            input_length = torch.full((B,), input_length, device=inputs.device, dtype=torch.long)

        jump_len = input_length
        current_state = state
        hidden_states = []

        for t in range(T):
            x_t      = inputs[:, t]
            is_valid = pad_mask[:, t]

            h_new_t, hidden_new, memory_new, pos_tape_new, usage_new, write_w_new = \
                self._step(x_t, current_state, jump_len)

            if self.use_lstm:
                new_h = (
                    torch.where(is_valid.unsqueeze(-1), hidden_new[0], current_state.hidden[0]),
                    torch.where(is_valid.unsqueeze(-1), hidden_new[1], current_state.hidden[1]),
                )
            else:
                new_h = torch.where(is_valid.unsqueeze(-1), hidden_new, current_state.hidden)

            new_mem = torch.where(is_valid.view(B, 1, 1, 1), memory_new,   current_state.memory)
            new_pos = torch.where(is_valid.view(B, 1, 1, 1), pos_tape_new, current_state.pos_tape)

            if self.use_soft_write:
                new_usage = torch.where(is_valid.unsqueeze(-1), usage_new, current_state.usage)
                new_ww    = torch.where(is_valid.view(B, 1, 1), write_w_new, current_state.write_w)
            else:
                new_usage = current_state.usage   # not tracked
                new_ww    = current_state.write_w  # not tracked

            current_state = VegStewState(
                memory=new_mem, hidden=new_h, pos_tape=new_pos,
                usage=new_usage, write_w=new_ww,
            )
            hidden_states.append(h_new_t)

        return torch.stack(hidden_states, dim=1), current_state
