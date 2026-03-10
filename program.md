# autoresearch

This is an experiment to have the LLM do its own research on the industrial-automaton
project. The goal is to autonomously improve model architectures and training strategies
to solve more of the benchmark tasks.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files for full context**:
   - `README.md` — repository overview.
   - `EXPERIMENTS.md` — all experiments run so far, best configs, consistency results.
   - `src/industrial_automaton/models/tape.py` — TapeRNN and BabyNTM implementations. **This is the primary file you will modify.**
   - `src/industrial_automaton/config.py` — all configurable CLI arguments.
   - `src/industrial_automaton/trainer.py` — training loop, loss function, evaluation. **You can modify this file too.**
4. **Initialize results.tsv**: Create `results.tsv` with just the header row (see format
   below). The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Editable surface

**What you CAN modify:**

- `src/industrial_automaton/models/tape.py` — the primary file. Modify `TapeRNN`,
  its controller, memory ops, head design, initialization. This is where all architecture
  experiments happen.
- `src/industrial_automaton/models/tape_config.py` (or add to `tape.py`) — new config
  dataclasses for new TapeRNN variants. Add to `config.py` if new CLI args are needed.
- `src/industrial_automaton/config.py` — add new config fields only. Do **not** remove
  or rename existing fields — other code depends on them.
- The CLI command in the experiment loop — all training and curriculum hyperparameters
  are passed as args to `inmaton`. Change them freely.

**What you CANNOT modify:**

- Any other model file (`baby_ntm`, `lstm`, `stack_rnn`, `transformer`, `implicit.py`,
  `common.py`) — leave them untouched. We are only iterating on TapeRNN.
- `src/industrial_automaton/tasks/` — task definitions and generators are fixed ground truth.
- `src/industrial_automaton/vocab.py` — the 55-token NSL vocabulary is frozen.
- `src/industrial_automaton/trainer.py` — the training loop, loss function, and evaluation
  harness are fixed. The `seq_acc` and `tok_acc` computed here are the ground truth metrics.
- `src/industrial_automaton/cli.py` — frozen, except registering a new TapeRNN variant
  in the model dispatch dict (one-line addition only).
- `src/industrial_automaton/generators.py` — data generation pipeline is frozen.
- Install new packages or add new dependencies. Use only what is in `pyproject.toml`.

## Hardware

Training runs on **Apple M3 Pro CPU** (no GPU). JAX uses the CPU backend. Each run
should complete within a **10-minute wall-clock budget**. If a run exceeds 10 minutes,
kill it and treat it as a crash.

## Running an experiment

The entry point is like this:

```bash
uv run inmaton \
    --run_name <tag> \ // same as tag from above. Ensure that assets/training_runs/<tag> does not exist.
    --task <task_name> \
    --model tape_rnn \
    --model_kwargs '{"hidden_size": 128, "memory_size": 16, "memory_cell_size": 8}' \
    --embedding_type learnable \
    --embedding_dim 16 \
    --learning_rate 1e-3 \
    --optimizer adam \
    --batch_size 64 \
    --dataset_size 1024 \
    --eval_dataset_size 256 \
    --max_steps 20000 \
    --eval_steps 500 \
    --early_stopping_patience 12 \
    --hard_array_limit 64 \
    --seed 4 \
    --log_level WARNING \
    --timeout 600 \  // 10-min timeout
    > run.log 2>&1
```

Redirect everything `> run.log 2>&1` - do NOT use tee or let output flood your context.

For curriculum training, add:

```bash
    --curriculum_type adaptive \
    --curriculum_kwargs '{"advance_threshold":0.90,"ema_decay":0.95,"advance_streak":3,"step_size":1,"min_bound":3,"max_bound":12}' \
```

## Goals and metric

**The goal: maximise `seq_acc`** (sequence accuracy — the entire output sequence must be
correct). `tok_acc` (token accuracy) is secondary and reported for context. **Do not**
change the loss function or evaluation harness.

**Model**: Always `tape_rnn`. The only model being iterated on is TapeRNN and variants
of it defined in `tape.py`. Do not switch to baby_ntm, lstm, or any other model.

**Task progression:**
- **Phase 1**: Optimise a single target task. Start with `associative_recall` (hardest
  reliably solved task, best baseline: tape_rnn seq_acc=0.89 at h=128, m=16, cell=8, L=8).
- **Phase 2**: Once Phase 1 task is consistently ≥ 0.95 across both seeds, add a second
  task. Run on both tasks and measure average seq_acc.
- **Phase 3**: Multi-task training

**Model size cap**: Keep total parameter count below **1 000 000 params** (1M). The model
param count is printed at startup as `Model params: X.XXX K`. Exceeding 1M is a soft
constraint — flag it in the description but don't automatically discard.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement
that adds ugly complexity is not worth it. Removing code and getting equal or better
results is a win. Weigh complexity cost against improvement magnitude.

**The first run**: Your very first run should always establish the baseline — run with
the best known config from `EXPERIMENTS.md` without any changes.

**OOD Generalisation**: All models when evaled on their training size work well, we want
models that can generalise to much longer sequence lengths, that is the real test. Keep
eval max seqlen atleast 20x longer than training.

## Seed policy

Results only count if they hold across **two seeds**.

- Perform first run. If it fails badly, discard immediately.
- If it looks promising (seq_acc ≥ 0.50), run seed=4 to confirm.
- A result is **consistent** if both seeds achieve seq_acc ≥ 0.50.
- A result is **flaky** if only one seed achieves ≥ 0.50 — note it but do not count it
  as a solved task.
- When logging to results.tsv, log the **lower** of the two seed results as seq_acc
  (conservative estimate). Add both individual values in the description.

`reverse_string` and `associative_recall` are the two reliably solved TapeRNN tasks.
`deduplicate_inputs` and `duplicate_string` are seed-sensitive and not reliably solved.
Print all the tasks using `uv run inmaton-tasks` for all the tasks.

## Output format

`inmaton` prints training progress and a final summary. Extract results with:

```bash
grep "^Final\|Model params\|Eval @" run.log | tail -5
```

The final lines look like:

```
Model params: 24.795 K
...
  Eval @ 20000: loss=0.1234 | tok_acc=0.9450 | seq_acc=0.8910
Training complete. 20000 steps | Tokens In: 12.3mn | Final Eval Acc: 0.8910
Final loss: 0.1234
Final token accuracy: 0.9450
Final sequence accuracy: 0.8910
```

Extract the key metrics:

```bash
grep -E "^Final (token|sequence)|Model params" run.log
```

Wall time is measured externally with `time ...`.

## Logging results

Log every run to `results.tsv` (tab-separated, NOT comma-separated).

**Do not commit `results.tsv` — leave it untracked by git.**

Header and columns:

```
commit	task	seq_acc	tok_acc	params_k	wall_mins	status	description	command
```

1. `commit` — git commit hash (short, 7 chars)
2. `task` — task name (e.g. `associative_recall`)
3. `seq_acc` — lower of the two seed results; 0.000000 for crashes
4. `tok_acc` — lower of the two seed results; 0.000000 for crashes
5. `params_k` — model param count in thousands (e.g. `24.8`)
6. `wall_mins` — total wall time in minutes for both seeds combined (e.g. `8.4`)
7. `status` — `keep`, `discard`, or `crash`
8. `description` — short description of what was tried (no tabs, no commas)
9. `command` — the full `uv run inmaton ...` command used

Example:

```
commit	task	seq_acc	tok_acc	params_k	wall_mins	status	description	command
a1b2c3d	associative_recall	0.762	0.850	24.8	8.4	keep	baseline tape_rnn h=128 m=16 cell=8	uv run inmaton --task associative_recall ...
b2c3d4e	associative_recall	0.831	0.912	24.8	8.6	keep	lower lr to 5e-4	uv run inmaton --task associative_recall --learning_rate 5e-4 ...
c3d4e5f	associative_recall	0.701	0.810	24.8	8.5	discard	switch to GeLU in controller	uv run inmaton ...
d4e5f6g	associative_recall,duplicate_string	0.000	0.000	0.0	0.0	crash	new model OOM on M3 Pro	uv run inmaton ...
```

## The experiment loop

The experiment runs on the dedicated autoresearch branch.

**LOOP FOREVER:**

1. Read config.py to see if there are any new hyperparameters.
2. Check git state: `git log --oneline -5` and `git status`.
3. Form a hypothesis — what change might improve seq_acc and why?
4. Make the change: edit model files, config.py, or the CLI args.
5. `git commit -m "brief description of change"`
6. `time uv run inmaton [args] > run.log 2>&1`
7. If run exceeds **15 minutes**, kill it: `kill %1` — treat as crash, skip seed=4.
8. Parse results: `grep -E "^Final (token|sequence)|Model params" run.log`
9. If grep is empty → crash. Run `tail -50 run.log` for the traceback.
   - Simple fix (typo, missing import)? Fix and re-run.
   - Fundamentally broken? Log as crash, `git reset --hard HEAD~1`, move on.
10. If seq_acc ≥ 0.50 then run again with `seed=4`: `time uv run inmaton [args] --seed 4 > run_4.log 2>&1`
11. Record both results in `results.tsv` using the **lower** seq_acc.
12. **If seq_acc improved over previous best** (both seeds ≥ prev best):
    - Status = `keep`. Advance — stay on this commit.
13. **If seq_acc did not improve**:
    - Status = `discard`. `git reset --hard HEAD~1`.
14. Repeat.

**Crashes**: Fix obvious bugs and retry once. If the idea is fundamentally broken, log
`crash`, reset, move on.

**Stuck?** Re-read `EXPERIMENTS.md` and `src/industrial_automaton/models/tape.py`. Ideas
to try within TapeRNN:
- Different controller (VanillaRNN → LSTM → GRU)
- Different head design (how read/write weights are computed)
- Different memory cell size or number of memory slots
- Different initialization (weight init, tape init)
- Additional tape ops or modified movement logic
- Curriculum learning (adaptive L=3→max) for tasks where tok_acc >> seq_acc
- Lower learning rate or different optimizer (adam → adamw → sgd)
- Combining near-miss configs from `EXPERIMENTS.md`

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human whether to
continue. The human may be away. Run autonomously until manually interrupted. If you
run out of ideas, think harder — re-read the code, revisit failed experiments, try more
radical changes. The loop runs until the human stops you.

On M3 Pro each run takes roughly 3–10 minutes depending on model size and max_steps.
At ~15 min per experiment pair (both seeds), you can run ~4 experiments/hour.

As an example use case, a user might leave you running while they sleep. If each experiment
takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the
duration of the average human sleep. The user then wakes up to experimental results, all
completed by you while they slept!

