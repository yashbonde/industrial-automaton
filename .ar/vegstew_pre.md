# VegStew — Pre-Implementation Design

VegStew is a hybrid of Tallerman and DNC that keeps what works from each and resolves the conflicts.

---

## Design Principles

The two architectures solve the same problems in different ways. The merge rule: **keep the mechanism that is cheaper, more stable, or proven better in the mar11 ablations. Where DNC solves a problem Tallerman doesn't (allocation, k/v split), add the lightweight version of it.**

---

## Memory Cell Layout

**Change from both**: Split each memory cell into a **key half** and **value half**.

```
cell_size = key_size + val_size   (e.g. 8 + 8 = 16)
memory: (B, num_heads, N, cell_size)
        |___________|  |__________|
          key half       val half
```

This directly addresses the associative recall ceiling. Content attention queries against the key half and reads out the value half. The model can learn "store the key here, store the value there" without them interfering.

---

## Read Mechanisms

| Mechanism | DNC | Tallerman | VegStew | Why |
|-----------|-----|-----------|---------|-----|
| Windowed local read | ❌ | ✅ tanh + W_m, centered at head | ✅ keep | Cheap O(window), proven essential for local tasks |
| Content-based read | ✅ cosine sim, separate read keys + strengths | ✅ dot-product h_t @ W_q on full cell | ✅ dot-product, query **key half only**, read **value half** | Mar11: dot-product better calibrated than cosine; k/v split fixes assoc ceiling |
| Positional attention | ❌ | ✅ sinusoidal pos_tape, h_t @ W_pq query | ✅ keep | Replaces DNC temporal links at O(N·pos_dim) vs O(N²) |
| Temporal link traversal | ✅ forward/backward via N×N linkage matrix | ❌ | ❌ drop | pos_tape + tape roll achieves same sequential traversal without O(N²) state |
| Read mode mixing | ✅ 3-way softmax (content/fwd/bwd) | additive sum | additive sum | Additive is simpler, gradients flow to all paths; modes are now structurally non-overlapping |
| Read key/strength sharpening | ✅ learned per-head key + scalar strength β | scalar 1/√cell_size | learned scalar β per head (sigmoid-gated) | DNC's sharpening lets the model go sharper or softer; Tallerman's fixed scale was shown optimal but per-head β is +0 params and more expressive |

---

## Write Mechanisms

| Mechanism | DNC | Tallerman | VegStew | Why |
|-----------|-----|-----------|---------|-----|
| Write address selection | ✅ content-based + allocation-based, interpolated by `allocation_gate` | ❌ always slot 0 (current head) | ✅ keep DNC style — soft weights over all N slots | Slot-0-only is what causes Tallerman to depend on roll actions being perfect; soft write enables gradient to reach any slot |
| Write value network | linear from controller | ✅ 3-layer MLP, LayerNorm + GELU | ✅ keep Tallerman MLP | MLP write is strictly more expressive; linear is a special case |
| Erase | ✅ per-word erase vector `e_t` ∈ (0,1)^W per write head | scalar erase gate on slot 0 only | ✅ per-word erase vector (DNC) | Scalar erase can't selectively blank part of a cell; per-word is essential when key/value halves should erase independently |
| Write gate | ✅ overall scalar write gate | ✅ same | ✅ keep | Identical in both — keep as-is |
| Allocation gate | ✅ interpolates content-write vs allocation-write | ❌ | ✅ keep | Without allocation, model overwrites live memory |
| Usage / freeness tracking | ✅ full freeness module: write increases usage, free_gate decreases it | ❌ | ✅ simplified: usage = EMA of `\|write_weights\|`, no explicit free_gate | Full DNC freeness needs `free_gate` from controller which adds interface params; EMA usage achieves same effect with less surface area |
| Input write residual | ❌ | ✅ `use_input_write`: x_t @ W_xi.T added to write vector | ✅ keep as optional flag | Proven to solve dedup (direct token hash); harmful for assoc — keep as config option |

---

## Memory Navigation

| Mechanism | DNC | Tallerman | VegStew | Why |
|-----------|-----|-----------|---------|-----|
| Tape roll actions | ❌ | ✅ stay/left/right/jump+/jump- (5-way soft action) | ✅ keep | Sequentially biased tasks (n_back, repeat_copy) need this; DNC has no concept of "current position" |
| Positional tape | ❌ | ✅ sinusoidal PE co-rolled with memory | ✅ keep | Zero extra parameters; gives positional attention a grounding signal |
| Temporal link graph | ✅ N×N per write head | ❌ | ❌ drop | O(N²) state; pos_tape + roll achieves the same traversal cheaper |
| Write head location | implicit (content/alloc weights) | tape head = slot 0 | **hybrid**: soft write weights over all N, modulated by a Gaussian locality prior centered at current head position | Pure DNC loses locality; pure Tallerman loses flexibility. Locality-modulated write keeps the head as a "center of mass" without forcing it |

The locality prior: before normalizing write weights, multiply by `exp(-d² / 2σ²)` where `d` is the slot distance from the current head position and `σ` is a learned scalar per head. When `σ→∞` this recovers DNC; when `σ→0` it recovers Tallerman.

---

## Controller

| Aspect | DNC | Tallerman | VegStew | Why |
|--------|-----|-----------|---------|-----|
| Cell type | LSTM fixed | configurable: vanilla / GRU / LSTM | ✅ keep configurable | Mar11: GRU optimal for assoc/dedup, LSTM for n_back/repeat_copy |
| Stability | scalar `clip_value` | ✅ LayerNorm + residual `LN(h_new + h_prev)` | ✅ keep LayerNorm + residual | Clipping destroys gradient magnitude information; LN doesn't |
| Controller input | `[x_t; prev_read_output]` | `[x_t; mem_read + content_read + pos_read]` | `[x_t; windowed_read + content_read + pos_read]` | All reads summed before concat to keep input dim fixed |
| Output projection | linear to `output_size` | direct `h_t` to logits | direct `h_t` | Tallerman's approach is leaner; DNC linear only needed because it concatenates `[h_t; read_words]` |

---

## Interface Signals (what controller emits per step)

| Signal | DNC | Tallerman | VegStew |
|--------|-----|-----------|---------|
| Write vector | ✅ linear | ✅ 3-layer MLP | ✅ MLP |
| Erase vector | ✅ sigmoid linear | scalar gate | ✅ sigmoid linear (per word, per head) |
| Write gate | ✅ | ✅ | ✅ |
| Allocation gate | ✅ | ❌ | ✅ |
| Free gate | ✅ | ❌ | ✅ simplified (EMA usage, no explicit free gate) |
| Read key | ✅ separate linear | implicit (h_t @ W_q) | ✅ dedicated W_q for key-half query |
| Read strength β | ✅ | fixed 1/√cell | ✅ learned scalar per head |
| Tape action logits | ❌ | ✅ h_t @ W_a | ✅ keep |
| Write locality σ | ❌ | ❌ | ✅ learned scalar per head (new) |

---

## State

| Component | DNC | Tallerman | VegStew |
|-----------|-----|-----------|---------|
| Memory | `(N, W)` | `(heads, N, cell_size)` | `(heads, N, key_size + val_size)` |
| Positional tape | ❌ | `(heads, N, pos_dim)` sinusoidal | `(heads, N, pos_dim)` sinusoidal |
| Read weights | `(num_reads, N)` | — | — (content attn is stateless) |
| Write weights | `(num_writes, N)` | — | `(heads, N)` kept for usage computation |
| Linkage matrix | `(num_writes, N, N)` | ❌ | ❌ dropped |
| Usage | `(N,)` | ❌ | `(N,)` EMA-based |
| Controller | LSTM (h, c) | RNN/GRU/LSTM | RNN/GRU/LSTM |

**Memory scaling**: DNC `O(N·W + N²·num_writes)`. Tallerman `O(N·cell + N·pos_dim)`. VegStew `O(N·cell + N·pos_dim + N)` — Tallerman scaling, no quadratic.

---

## Forward Pass Per Step

```
1.  Windowed read:      window slots around head → tanh(W_m) → mem_read          [Tallerman]
2.  Content read:       h_t @ W_q → query key-half → softmax(β·scores) → val-half [hybrid k/v + β]
3.  Pos attention:      h_t @ W_pq → query pos_tape → softmax → read val-half     [Tallerman]
4.  Controller:         [x_t; mem_read + content_read + pos_read] → h_new          [Tallerman]
5.  LN + residual:      h_new = LN(h_new + h_prev)                                 [Tallerman]
6.  Write value:        MLP(h_new) [+ optional x_t @ W_xi residual]                [Tallerman]
7.  Erase vector:       sigmoid(W_e h_new) ∈ (0,1)^cell_size                       [DNC]
8.  Write address:      content_w*(1-alloc_gate) + alloc_w*alloc_gate,
                        × Gaussian locality prior around head pos                   [hybrid]
9.  Update usage:       usage = EMA(usage, sum(write_weights, dim=slot))            [simplified DNC]
10. Write to memory:    M *= (1 - w⊗e);  M += w⊗v                                 [DNC erase+add]
11. Tape actions:       h_new @ W_a → softmax → stay/left/right/jump+/jump-        [Tallerman]
12. Roll memory + pos_tape by action weights                                        [Tallerman]
```

---

## What Got Dropped Entirely

| Feature | From | Reason |
|---------|------|--------|
| Temporal linkage matrix | DNC | O(N²) state; pos_tape + roll is a cheaper equivalent |
| Forward/backward read mode | DNC | Replaced by positional attention + tape roll |
| Explicit free_gate | DNC | EMA usage is simpler; free_gate adds interface params with marginal benefit |
| Flat interface projection | DNC | Dedicated linear layers (Tallerman style) are more interpretable and avoid bottleneck |
| Value clipping | DNC | LayerNorm + residual is strictly better for gradient flow |
| Fixed write-to-slot-0 | Tallerman | Too rigid; allocation needed to avoid overwriting live data |
| Scalar erase gate | Tallerman | Per-word erase needed for selective cell updates, especially with k/v split |

---

## Summary

| Dimension | DNC wins | Tallerman wins | VegStew solution |
|-----------|----------|----------------|------------------|
| Write addressing | ✅ content + allocation | | Soft write with locality modulation |
| K/V separation | | ❌ missing (assoc ceiling cause) | Split cell into key_size + val_size halves |
| Erase granularity | ✅ per-word vector | | Per-word erase vector |
| Read: local patterns | | ✅ windowed read | Windowed read kept |
| Read: positional | | ✅ pos_tape + pos_attn | Kept; replaces linkage |
| Temporal traversal | ✅ linkage | | Dropped; pos_tape + roll is O(N) equivalent |
| State scaling | | ✅ O(N·cell) | O(N·cell) — no linkage |
| Controller stability | | ✅ LN + residual | LN + residual |
| Controller flexibility | | ✅ GRU/LSTM/vanilla | Kept configurable |
| Input write residual | | ✅ use_input_write | Optional flag kept |
| Usage/allocation | ✅ freeness | | EMA usage (simplified) |
| Write value network | | ✅ 3-layer MLP GELU | MLP kept |
