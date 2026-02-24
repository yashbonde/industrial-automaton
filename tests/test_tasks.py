"""Stress tests for all task generators: every generator runs at max length without error."""

import importlib
import time

import numpy as np
import pytest


# Per-generator maximum valid length, respecting validation limits.
_MAX_LENGTHS = {
    "generate_binary_addition": 64,
    "generate_binary_multiplication": 32,
    "generate_square_root": 19,
    "generate_repeat_copy_n": 1000,
    "generate_duplicate_string": 3333,
    "generate_deduplicate_inputs": 3333,
    "generate_missing_duplicate": 5000,
    "generate_odds_first": 5000,
    "generate_associative_recall": 18,
    "generate_n_back": 4999,
    "generate_python_execution": 999,
    "generate_sort": 5000,
    "generate_stack_manipulation": 5000,
    "generate_reverse_string": 5000,
}

_TRIPLE_GENERATORS = {
    "generate_shortest_path",
    "generate_mst_prim",
    "generate_graph_traversal",
    "generate_tsp",
    "generate_convex_hull",
    "generate_delaunay",
}

BINARY_TASK_NAMES = {"generate_binary_addition", "generate_binary_multiplication"}

LARGE_LENGTH = 5000
BATCH_SIZE = 1

# Module mapping — binary tasks live inside context_sensitive
_MODULE_MAP = {
    "regular": "regular",
    "context_free": "context_free",
    "context_sensitive": "context_sensitive",
    "data_processing": "data_processing",
    "graphs_geometry": "graphs_geometry",
}


def _get_valid_length(gen_name, base_length):
    length = min(base_length, _MAX_LENGTHS.get(gen_name, base_length))
    if gen_name == "generate_modular_arithmetic":
        return length if length % 2 != 0 else length - 1
    if gen_name == "generate_count_n":
        return length - (length % 3)
    if gen_name in _TRIPLE_GENERATORS:
        return length - (length % 3)
    return length


def _collect_generators():
    """Discover all (gen_name, callable, lengths) across every task module."""
    cases = []
    for module_name in _MODULE_MAP.values():
        try:
            mod = importlib.import_module(f"industrial_automaton.tasks.{module_name}")
        except ImportError:
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("generate_"):
                continue
            fn = getattr(mod, name)
            if name == "generate_binary_addition":
                lengths = [8, 16, 32, 64]
            elif name == "generate_binary_multiplication":
                lengths = [8, 16, 32]
            else:
                lengths = [_get_valid_length(name, LARGE_LENGTH)]
            for length in lengths:
                cases.append((name, fn, length))
    return cases


_ALL_CASES = _collect_generators()


@pytest.mark.parametrize(
    "gen_name,generator,length",
    _ALL_CASES,
    ids=[f"{name}-len{l}" for name, _, l in _ALL_CASES],
)
def test_generator(gen_name, generator, length):
    result = generator(batch_size=BATCH_SIZE, length=length)

    assert "input" in result, f"{gen_name}: missing 'input' key"
    assert "output" in result, f"{gen_name}: missing 'output' key"
    assert result["input"].shape[0] == BATCH_SIZE
    assert not np.any(np.isnan(result["input"])), f"{gen_name}: NaN in input"
    assert not np.any(np.isnan(result["output"])), f"{gen_name}: NaN in output"
