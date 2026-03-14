# Mar14 Autoresearch — VegStew Single-Task Focus

## Session Setup
- Branch: autoresearch/mar14
- Budget: 2 hours max per experiment
- Scope: VegStew architecture only; target tasks: associative_recall, deduplicate_inputs, n_back, repeat_copy_n

## State from Previous Sessions

### Best known VegStew configs (from results.tsv + mar13 logs)
- **assoc**: cosine_attn=true, use_soft_write=false, m=40, k=8, v=8, w=3, pos_attn=true → 0.252 @ eval@20 (20k steps, 5min)
- **dedup / repeat_copy / n_back**: VegStew never properly benchmarked at long training (all mar13 runs were too short or OOD eval@100+)
- **Tallerman ceiling**: assoc=0.9141, dedup=1.000, n_back=0.8906, repeat_copy=1.000 (eval@20, adaptive curriculum, 30k+ steps)

### Key mar11 winning configs (Tallerman — targets to match/beat):
- assoc: h=256, m=60, gru, no posattn, w=5, uniform curriculum min=3 max=12
- n_back: h=256, m=60, lstm, posattn, w=3, adaptive curriculum
- dedup: h=256, m=100, gru, no posattn, w=1, use_input_write
- repeat_copy: h=256, m=60, lstm, posattn, w=3

### Key findings
1. cosine_attn is the single biggest VegStew win (+54% vs dot-product for assoc)
2. K/V split + cosine_attn synergy confirmed
3. Soft write (DNC) consistently hurts — simple write wins
4. 5-min per run was severely limiting for dedup/repeat_copy/n_back

## Experiment Plan

### Exp 1 — Long-run VegStew baselines on all 4 tasks (2h each, parallel)
Hypothesis: 5-min training was the bottleneck. With 2h and adaptive curriculum the VegStew cosine_attn config should approach/exceed Tallerman on all tasks.
Config: cosine_attn=true, use_soft_write=false, m=60, k=8, v=8, w=3, pos_attn=true, lstm, adaptive curriculum
Run all 4 tasks in parallel.

### Exp 2 — MoE Output Heads (K=4, soft router)
Hypothesis: From vegstew_02.md — single output head creates interference between tasks. For single-task this likely doesn't help, but for seq generation tasks (repeat_copy, dedup) which have highly structured output patterns, multiple specialized decoders could reduce gradient noise.

### Exp 3 — IRQ (Input-Read Query) for assoc/dedup
The use_input_read_query flag was added but never properly tested at long training. For assoc, reading from the memory using the current *input token* (not hidden state) as query key is the natural lookup mechanism.

### Exp 4 — Asymmetric K/V (k=16, v=8 or k=8, v=16)
From mar13 logs mar13_k8v16 and mar13_k16 experiments — which ratio helps which task?

## Results

(populated as experiments run)
