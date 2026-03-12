# autoresearch

This is an experiment to have the LLM do its own research on the industrial-automaton project. The goal is to autonomously improve model architectures and training strategies to solve more of the benchmark tasks.

You are a research scientist that is trying to invent new neural network architectures.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
   - If the branch exists then continue working on that branch.
   - Else, create the branch: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files for full context**:
   - `README.md` — repository overview.
   - `src/industrial_automaton/models_torch/tallerman.py` — **This is the primary file you will modify.**
   - `src/industrial_automaton/config.py` — all configurable CLI arguments.
   - `src/industrial_automaton/trainer_torch.py` — training loop, loss function, evaluation. **You can modify this file too.**
   - Do not edit 
4. **Initialize results.tsv**: Create `results.tsv` with just the header row (see format
   below). The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## What you CANNOT modify

- Any other model file (`baby_ntm`, `lstm`, `stack_rnn`, `transformer`, `implicit.py`,
  `common.py`) — leave them untouched. We are only iterating on Tallerman architecture.
- `src/industrial_automaton/tasks/` — task definitions and generators are fixed ground truth.
- `src/industrial_automaton/generators.py` — data generation pipeline is frozen.
- Install new packages or add new dependencies. Use only what is in `pyproject.toml`.
- **Do not** change the loss function or evaluation harness.

## Running an experiment

To get the latest arguments run `uv run inmaton -help`. Here's a few fixed arguments you must
honour:

```bash
uv run inmaton \
    --run_name <tag> \ // same as tag from above. Ensure that assets/training_runs/<tag> does not exist.
    --task <task_name> \
    --model tallerman \
    --timeout 300 \  // 5-min timeout, super critical
    [args] ...
    > logs/<tag>.log 2>&1
```

Redirect everything `> logs/run.log 2>&1` - do NOT use tee or let output flood your context.

For curriculum training, add:

```bash
    --curriculum_type adaptive \
    --curriculum_kwargs '{"advance_threshold":0.90,"ema_decay":0.95,"advance_streak":3,"step_size":1,"min_bound":3,"max_bound":12}' \
```

All the bash commands will be run in background (`run_in_background=true`) and then create a wait task on the jobs. You can try to be ambitious and run several bash in parallel and a single wait task to wait for all of them to finish.

## Goals and metrics

**The goal**: Train the smallest memory based model to maximise eval `seq_acc` (sequence accuracy — the entire output sequence must be correct) on `repeat_copy_n, deduplicate_inputs, associative_recall, n_back` tasks.

To achieve the goal everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes (or shows signs of promise) within the time budget.

- **OOD Generalisation**: Keep eval max seqlen atleast 20x longer than training. Minimum train length is 20 and max eval length is 400.

**Model size cap**: Keep total parameter count below 1Mn. The model param count is printed at startup as `Model params: X.XXX K`. Exceeding 1Mn is a soft constraint — flag it in the description but don't automatically discard.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win. Weigh complexity cost against improvement magnitude.

**The first run**: Your very first run should always establish the baseline — run with the best known config from `results.tsv`.

## Output format

`inmaton` prints training progress and a final summary. Extract results with:

```bash
grep "^Final\|Model params\|Eval @" logs/run.log | tail -5
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
grep -E "^Final (token|sequence)|Model params" logs/run.log
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
2. `task` — task name(s) with comma (e.g. `associative_recall,deduplicate_inputs`)
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

All the runs must be logged here, even the crashed one.

## The experiment loop

The experiment runs on the dedicated autoresearch branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

**LOOP FOREVER:**

1. Read config.py, program.md to see if there are any new hyperparameters, changes in instruction.
2. Check git state: `git log --oneline -5` and `git status`. Read few latest experiments in the `./.autoresearch/` folder. Form a hypothesis — what is known from previous experiments, what change might improve seq_acc and why? Create an experiment one pager and save in `./.autoresearch/<tag>.md` file.
3. Make the change: edit model files, config.py, or the CLI args. Then `git commit -m "brief description of change"`
4. Train the model and Parse results: `grep -E ...`
   - If run exceeds timeout / If empty / Fundamentally broken — mark as crash in results
   - Simple fix (typo, missing import)? Fix and re-run.
   - If seq_acc > 0.8 then run again with `--seed=4` to ensure we didn't hit a lucky seed
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If seq_acc improved (higher), you "advance" the branch, keeping the git commit
9. If seq_acc is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: Fix obvious bugs and retry once. If the idea is fundamentally broken, log `crash`, reset, move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers (explore agent in `../papers/`) referenced in the code (`../`), re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
