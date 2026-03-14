## Previous Results - Mar13

Results of mar13_all24 — 17.5 hours total, 24 tasks, VegStew (LSTM + cosine + IRQ + soft_write, memory=80)

┌────────────────────┬─────────┬───────────────────────────────────────────────────────┐
│        task        │ seq_acc │                        status                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ even_pairs         │ 0.653   │ ↓ regressed from peak 0.996                           │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ dyck_n             │ 0.680   │ ↓ regressed from peak 0.862                           │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ parity_check       │ 0.688   │ ↓ regressed from peak 0.783                           │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ cycle_navigation   │ —       │ peaked ~0.689 at 15.5h                                │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ count_n            │ —       │ peaked ~0.773 at 15.5h                                │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ modular_arithmetic │ 0.574   │ stable                                                │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ nested_mod_arith   │ 0.556   │ stable                                                │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ n_back             │ 0.259   │ slowly improving                                      │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ tsp                │ 0.163   │ slowly improving                                      │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ reverse_string     │ 0.073   │ newly appeared at 17.5h                               │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ stack_manipulation │ 0.032   │ newly appeared at 17.5h                               │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ repeat_copy_n      │ 0.056   │ barely moving                                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ deduplicate_inputs │ 0.085   │ barely moving                                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ missing_duplicate  │ 0.175   │ slowly improving                                      │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ associative_recall │ ~0.367  │ below single-task ceiling                             │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ shortest_path      │ 0.068   │ barely moving                                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ odds_first         │ 0.062   │ barely moving                                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ convex_hull        │ ~0.109  │ slowly improving                                      │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ delaunay           │ 0.030   │ barely moving                                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ duplicate_string   │ 0.055   │ barely moving                                         │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ graph_traversal    │ 0.000   │ completely stuck (tok_acc=0.61 — local patterns only) │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ mst_prim           │ 0.000   │ completely stuck (tok_acc=0.51)                       │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ sort               │ 0.000   │ completely stuck (tok_acc=0.11)                       │
├────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ mini_shrdlu        │ 0.015   │ near zero                                             │
└────────────────────┴─────────┴───────────────────────────────────────────────────────┘

Key findings:

1. 9 tasks are meaningfully learning — the model can handle highly diverse algorithmic tasks in a single set of weights, which is itself a strong result from scratch.
2. Catastrophic forgetting is the dominant failure mode — easy tasks (even_pairs 0.996 → 0.653, dyck_n 0.862 → 0.680) regress as the adaptive sampler shifts budget toward harder tasks.
min_weight=0.15 is insufficient to protect learned tasks.
3. Three tasks are architecturally stuck — graph_traversal, mst_prim, and sort have high tok_acc but zero seq_acc, meaning the model learns local token statistics but not the
underlying algorithm. These likely need either more memory capacity or the MoE output head.
4. The adaptive sampler works — tasks with 0 acc at 3.5h (tsp, reverse_string, stack_manipulation) started moving by 15.5h, confirming gradient budget reallocation is effective.
5. ZERO think steps were not yet applied — all 17.5h ran without the thinking buffer, so the output-heavy tasks (repeat_copy, reverse_string, dedup) were running under the
corrupted-memory regime. The new baseline with THINK_STEPS=3 should directly help these.

## New experiments

### ZERO Think Steps (mandatory baseline)

The hypothesis is that during output generation the model continues writing to memory using the new input tokens (the output tokens it just emitted), progressively corrupting the stored input representation that it needs to re-read for tasks like repeat_copy_n, reverse_string, and deduplicate_inputs. By inserting 3 ZERO tokens between YIELD and the first output token — with no loss computed on these steps — the model gets dedicated consolidation time where it can learn to close its write gates and stabilize the memory state before generation begins. This is a mandatory architectural baseline applied to all experiments, implemented as a single THINK_STEPS=3 constant in generators.py that modifies _format_examples to insert [ZERO x 3] into every sequence at the YIELD boundary.

### MoE Output Heads (Exp 2)
The hypothesis is that a single shared output projection forces all 24 tasks to compete for the same decoder weights, creating destructive interference — the gradient signal for "predict the next BFS node" directly conflicts with "predict the parity bit." By replacing the single output head with K=4 learned heads and a soft router (a small linear layer on the hidden state feeding a softmax), the model can self-organize its decoders by output structure without any hard task-specific assignments. The routing is emergent: binary classification tasks (even_pairs, parity_check) will naturally cluster to different heads than sequence generation tasks (sort, repeat_copy_n). This removes output-layer interference while keeping the memory and LSTM fully shared for cross-task transfer.

###  Progressive Curriculum (Exp 3)
The hypothesis is that throwing all 24 tasks at the model from step 0 is computationally wasteful — easy tasks (even_pairs, parity_check) consume a disproportionate share of early gradient budget before they are solved, and the adaptive sampler can only redirect budget after the reweight interval fires. A progressive curriculum starts with the N easiest tasks and automatically expands the task pool when the current set exceeds a performance threshold, so easy tasks get solved cheaply with concentrated gradient signal and then fall to the min_weight floor, freeing capacity for harder tasks earlier. This is implemented as a new ProgressiveCurriculum type in the curriculum module, parameterized by threshold, add_n, and task_order (difficulty-ranked task list), and hooks into the AdaptiveSampler's reweight cycle so expansion happens automatically without manual intervention.
