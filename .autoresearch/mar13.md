# mar13 — VegStew Experiment Session

**Branch**: `autoresearch/mar13`
**Model**: VegStew (new architecture — hybrid of Tallerman + DNC)
**Goal**: Beat mar11 Tallerman baselines on all 4 tasks, especially associative_recall (ceiling at 0.9141)

---

## Context from mar11

| task | tallerman seq_acc | ceiling hit? |
|------|------------------|--------------|
| associative_recall | 0.9141 | YES — every variation tried |
| deduplicate_inputs | 1.0000 | Solved |
| n_back | 0.8906 | Soft ceiling (seed instability) |
| repeat_copy_n | 1.0000 | Solved |

Primary hypothesis for ceiling on assoc: **no K/V split in memory**. Content attention can't cleanly distinguish "this is the key I'm looking for" from "this is the value I want to return".

---

## VegStew Architecture (new in this session)

VegStew implements the design from `.ar/vegstew_pre.md`:
- Memory cell = key_size + val_size (e.g. 8+8=16)
- Content attention: query key-half, read val-half (fixes assoc ceiling)
- DNC-style write: content_w*(1-alloc_gate) + alloc_w*alloc_gate
- Gaussian locality prior centered at slot 0 (learned sigma per head)
- Per-word erase vector (vs Tallerman's scalar erase gate)
- EMA usage tracking for allocation
- Learned beta (sharpening) per head

---

## Experiment Plan

### Exp 1 — VegStew baseline on associative_recall
Hypothesis: K/V split breaks the 0.9141 ceiling on assoc.
Config: mirror the best tallerman assoc config (h=256, m=60, cell=8+8=16, gru, no posattn, w=5, curriculum).
Expected: >0.9141 seq_acc.

### Exp 2 — VegStew baseline on n_back
Config: mirror tallerman n_back (h=256, m=60, lstm, posattn, w=3).
Expected: ≥0.8906 (ideally better given better write mechanism).

### Exp 3 — VegStew on deduplicate_inputs
Config: mirror tallerman dedup (h=256, m=100, gru, no posattn, w=1, use_input_write).
Expected: 1.0000 maintained.

### Exp 4 — VegStew on repeat_copy_n
Config: mirror tallerman repeat_copy (h=256, m=60, lstm, posattn, w=3).
Expected: 1.0000 maintained.

---

## Results Log

(populated as experiments run)
