"""Tests for unified vocabulary: every task emits valid unified token IDs."""

import numpy as np
import pytest

from industrial_automaton import vocab as V
from industrial_automaton import tasks


# Task name -> (generator_fn, kwargs)
# We use small sizes to keep tests fast.
_TASK_CONFIGS = {
    "even_pairs":              (tasks.generate_even_pairs, dict(batch_size=4, length=6, vocab_size=2)),
    "parity_check":            (tasks.generate_parity_check, dict(batch_size=4, length=6, vocab_size=2)),
    "cycle_navigation":        (tasks.generate_cycle_navigation, dict(batch_size=4, length=6, num_states=5)),
    "modular_arithmetic":      (tasks.generate_modular_arithmetic, dict(batch_size=4, length=5, modulus=5)),
    "dyck_n":                  (tasks.generate_dyck_n, dict(batch_size=4, length=8, n=2)),
    "reverse_string":          (tasks.generate_reverse_string, dict(batch_size=4, length=6, vocab_size=8)),
    "stack_manipulation":      (tasks.generate_stack_manipulation, dict(batch_size=4, length=8, vocab_size=4)),
    "nested_modular_arithmetic": (tasks.generate_nested_modular_arithmetic, dict(batch_size=4, length=10, modulus=5)),
    "solve_equation":          (tasks.generate_solve_equation, dict(batch_size=4, length=7, max_val=9)),
    "binary_addition":         (tasks.generate_binary_addition, dict(batch_size=4, length=8)),
    "binary_multiplication":   (tasks.generate_binary_multiplication, dict(batch_size=4, length=8)),
    "duplicate_string":        (tasks.generate_duplicate_string, dict(batch_size=4, length=6, vocab_size=8)),
    "repeat_copy_n":           (tasks.generate_repeat_copy_n, dict(batch_size=4, length=5, max_n=3, vocab_size=8)),
    "odds_first":              (tasks.generate_odds_first, dict(batch_size=4, length=6, vocab_size=10)),
    "sort":                    (tasks.generate_sort, dict(batch_size=4, length=6, max_value=20)),
    "square_root":             (tasks.generate_square_root, dict(batch_size=4, length=3, base=10)),
    "count_n":                 (tasks.generate_count_n, dict(batch_size=4, length=6, n=3)),
    "n_back":                  (tasks.generate_n_back, dict(batch_size=4, length=8, vocab_size=8)),
    "associative_recall":      (tasks.generate_associative_recall, dict(batch_size=4, length=10, vocab_size=8)),
    "missing_duplicate":       (tasks.generate_missing_duplicate, dict(batch_size=4, length=5)),
    "deduplicate_inputs":      (tasks.generate_deduplicate_inputs, dict(batch_size=4, length=4, vocab_size=8)),
    "python_execution":        (tasks.generate_python_execution, dict(batch_size=4, length=10, max_val=9)),
    "mini_shrdlu":             (tasks.generate_mini_shrdlu, dict(batch_size=4, length=20, grid_size=3, num_blocks=4)),
    "shortest_path":           (tasks.generate_shortest_path, dict(batch_size=2, length=18, num_nodes=5, max_weight=9)),
    "mst_prim":                (tasks.generate_mst_prim, dict(batch_size=2, length=18, num_nodes=5, max_weight=9)),
    "graph_traversal":         (tasks.generate_graph_traversal, dict(batch_size=2, length=18, num_nodes=5)),
    "tsp":                     (tasks.generate_tsp, dict(batch_size=2, length=15, num_cities=5, coord_scale=50)),
    "convex_hull":             (tasks.generate_convex_hull, dict(batch_size=2, length=15, num_points=5, coord_scale=50)),
    "delaunay":                (tasks.generate_delaunay, dict(batch_size=2, length=15, num_points=5, coord_scale=50)),
}


@pytest.mark.parametrize("task_name", sorted(_TASK_CONFIGS.keys()))
def test_all_token_ids_in_range(task_name):
    """Every token ID in input/output must be < V.SIZE."""
    gen_fn, kwargs = _TASK_CONFIGS[task_name]
    rng = np.random.default_rng(42)
    result = gen_fn(**kwargs, rng=rng)

    inp = result["input"]
    out = result["output"]
    assert np.all(inp >= 0) and np.all(inp < V.SIZE), \
        f"{task_name} input has tokens outside [0, {V.SIZE}): min={inp.min()}, max={inp.max()}"
    assert np.all(out >= 0) and np.all(out < V.SIZE), \
        f"{task_name} output has tokens outside [0, {V.SIZE}): min={out.min()}, max={out.max()}"


@pytest.mark.parametrize("task_name", sorted(_TASK_CONFIGS.keys()))
def test_output_tokens_match_mask(task_name):
    """Output tokens must be a subset of the output_mask True positions."""
    gen_fn, kwargs = _TASK_CONFIGS[task_name]
    rng = np.random.default_rng(42)
    result = gen_fn(**kwargs, rng=rng)

    # Get vocab info
    vi_fn = tasks.VOCAB_INFO[task_name]
    # Extract task-specific kwargs for vocab info
    vi_kwargs = {}
    for key in ("vocab_size", "modulus", "num_states", "max_val", "n", "max_n",
                "base", "max_value", "num_blocks", "length"):
        if key in kwargs:
            vi_kwargs[key] = kwargs[key]
    info = vi_fn(**vi_kwargs)
    mask = info["output_mask"]

    out = result["output"].flatten()
    for tok in np.unique(out):
        tok = int(tok)
        assert mask[tok], \
            f"{task_name}: output token {tok} ({V.decode([tok])}) not in output_mask"


@pytest.mark.parametrize("task_name", sorted(_TASK_CONFIGS.keys()))
def test_input_tokens_in_input_set(task_name):
    """Non-PAD input tokens must be in the declared input_tokens set."""
    gen_fn, kwargs = _TASK_CONFIGS[task_name]
    rng = np.random.default_rng(42)
    result = gen_fn(**kwargs, rng=rng)

    vi_fn = tasks.VOCAB_INFO[task_name]
    vi_kwargs = {}
    for key in ("vocab_size", "modulus", "num_states", "max_val", "n", "max_n",
                "base", "max_value", "num_blocks", "length"):
        if key in kwargs:
            vi_kwargs[key] = kwargs[key]
    info = vi_fn(**vi_kwargs)
    valid = info["input_tokens"] | {V.PAD}  # PAD is always allowed in input

    inp = result["input"].flatten()
    for tok in np.unique(inp):
        tok = int(tok)
        assert tok in valid, \
            f"{task_name}: input token {tok} ({V.decode([tok])}) not in input_tokens"


def test_every_task_has_vocab_info():
    """Every task in TASK_NAMES has an entry in VOCAB_INFO."""
    for name in V.TASK_NAMES:
        assert name in tasks.VOCAB_INFO, f"Missing VOCAB_INFO for task '{name}'"


def test_vocab_size():
    """Vocab size constant is 200."""
    assert V.SIZE == 200


def test_task_token_mapping():
    """task_token returns unique IDs in the TASK range."""
    seen = set()
    for name in V.TASK_NAMES:
        tok = V.task_token(name)
        assert V.TASK_0 <= tok < V.TASK_0 + 36, f"task_token({name})={tok} out of range"
        assert tok not in seen, f"Duplicate task token {tok}"
        seen.add(tok)


def test_decode_roundtrip():
    """decode produces human-readable strings for all special tokens."""
    tokens = [0, 1, 50, 99, V.PAD, V.BOS, V.EOS, V.SEP,
              V.OP_ADD, V.OP_SUB, V.OP_MUL, V.OP_EQ,
              V.OPEN(0), V.CLOSE(0), V.VAR_X, V.OPEN_PAREN, V.CLOSE_PAREN,
              V.STACK_POP, V.NAV_FWD, V.NAV_BWD, V.HASH_SEP,
              V.TRUE, V.FALSE,
              V.PROG_ASSIGN, V.PROG_LOOP_START, V.PROG_LOOP_END,
              V.REL_ABOVE, V.REL_BELOW, V.REL_LEFT, V.REL_RIGHT]
    decoded = V.decode(tokens)
    assert len(decoded) == len(tokens)
    # No UNK tokens
    assert all("<UNK" not in d for d in decoded), f"UNK in decoded: {decoded}"
