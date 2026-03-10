"""Task generators and registry for neural automata training.

This module provides:
- MASTER_REGISTRY: Registry of all task generators with metadata
- Individual task functions exported for backward compatibility
- Utility functions: get_task(), list_tasks(), list_tasks_by_category()
"""

from typing import Optional, List, Dict, Any
from .registry import MASTER_REGISTRY
from .generators import generate_variable_dataset, create_batch_iterator

# ============================================================================
# Export individual task functions for backward compatibility
# ============================================================================

# Regular tasks
generate_even_pairs = MASTER_REGISTRY.even_pairs.fn
generate_parity_check = MASTER_REGISTRY.parity_check.fn
generate_cycle_navigation = MASTER_REGISTRY.cycle_navigation.fn
generate_modular_arithmetic = MASTER_REGISTRY.modular_arithmetic.fn

# Context-free tasks
generate_stack_manipulation = MASTER_REGISTRY.stack_manipulation.fn
generate_reverse_string = MASTER_REGISTRY.reverse_string.fn
generate_nested_modular_arithmetic = MASTER_REGISTRY.nested_modular_arithmetic.fn
from industrial_automaton.tasks.registry import generate_solve_equation  # commented out of registry
generate_dyck_n = MASTER_REGISTRY.dyck_n.fn

# Context-sensitive tasks
generate_duplicate_string = MASTER_REGISTRY.duplicate_string.fn
generate_repeat_copy_n = MASTER_REGISTRY.repeat_copy_n.fn
generate_deduplicate_inputs = MASTER_REGISTRY.deduplicate_inputs.fn
generate_associative_recall = MASTER_REGISTRY.associative_recall.fn
generate_missing_duplicate = MASTER_REGISTRY.missing_duplicate.fn
generate_odds_first = MASTER_REGISTRY.odds_first.fn
generate_count_n = MASTER_REGISTRY.count_n.fn
generate_n_back = MASTER_REGISTRY.n_back.fn

# Arithmetic tasks
generate_square_root = MASTER_REGISTRY.square_root.fn
# Binary arithmetic tasks are commented out in registry (not in NSL TASK_NAMES)
from industrial_automaton.tasks.registry import (
    generate_8_bit_addition,
    generate_16_bit_addition,
    generate_32_bit_addition,
    generate_64_bit_addition,
    generate_8_bit_multiplication,
    generate_16_bit_multiplication,
    generate_32_bit_multiplication,
)

# Data processing tasks
generate_sort = MASTER_REGISTRY.sort.fn
from industrial_automaton.tasks.registry import generate_python_execution  # commented out of registry
generate_mini_shrdlu = MASTER_REGISTRY.mini_shrdlu.fn

# Graphs & geometry tasks
generate_shortest_path = MASTER_REGISTRY.shortest_path.fn
generate_mst_prim = MASTER_REGISTRY.mst_prim.fn
generate_graph_traversal = MASTER_REGISTRY.graph_traversal.fn
generate_tsp = MASTER_REGISTRY.tsp.fn
generate_convex_hull = MASTER_REGISTRY.convex_hull.fn
generate_delaunay = MASTER_REGISTRY.delaunay.fn

# ============================================================================
# Convenience functions
# ============================================================================

def get_task(task_name: str):
    """Get task entry from registry.

    Args:
        task_name: Name of the task (without 'generate_' prefix)

    Returns:
        TaskEntry with function, metadata, and parameter presets

    Example:
        >>> task = get_task("even_pairs")
        >>> task.fn(rng, batch_size=32, **task.baseline)
    """
    if task_name not in MASTER_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' not found. Available tasks: {list(MASTER_REGISTRY.keys())}"
        )
    return MASTER_REGISTRY[task_name]


def list_tasks() -> List[str]:
    """List all registered task names.

    Returns:
        Sorted list of task names
    """
    return sorted(MASTER_REGISTRY.keys())


def list_tasks_by_category() -> Dict[str, List[str]]:
    """List tasks grouped by category.

    Returns:
        Dictionary mapping category names to lists of task names
    """
    result = {}
    for task_name, entry in MASTER_REGISTRY.items():
        if entry.category not in result:
            result[entry.category] = []
        result[entry.category].append(task_name)

    # Sort task names within each category
    for category in result:
        result[category].sort()

    return result


def get_task_info(task_name: str) -> Dict[str, Any]:
    """Get human-readable information about a task.

    Args:
        task_name: Name of the task

    Returns:
        Dictionary with task metadata
    """
    entry = get_task(task_name)
    return {
        "name": task_name,
        "category": entry.category,
        "baseline_params": entry.baseline,
        "max_params": entry.max,
        "papercode_params": entry.in_papercode,
        "min_length": entry.min_length,
        "docstring": entry.fn.__doc__,
    }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Registry and metadata
    "MASTER_REGISTRY",
    # Utility functions
    "get_task",
    "list_tasks",
    "list_tasks_by_category",
    "get_task_info",
    "generate_variable_dataset",
    "create_batch_iterator",
    # Individual task functions (backward compatibility)
    "generate_even_pairs",
    "generate_parity_check",
    "generate_cycle_navigation",
    "generate_modular_arithmetic",
    "generate_stack_manipulation",
    "generate_reverse_string",
    "generate_nested_modular_arithmetic",
    "generate_solve_equation",
    "generate_dyck_n",
    "generate_duplicate_string",
    "generate_repeat_copy_n",
    "generate_deduplicate_inputs",
    "generate_associative_recall",
    "generate_missing_duplicate",
    "generate_odds_first",
    "generate_count_n",
    "generate_n_back",
    "generate_square_root",
    "generate_8_bit_addition",
    "generate_16_bit_addition",
    "generate_32_bit_addition",
    "generate_64_bit_addition",
    "generate_8_bit_multiplication",
    "generate_16_bit_multiplication",
    "generate_32_bit_multiplication",
    "generate_sort",
    "generate_python_execution",
    "generate_mini_shrdlu",
    "generate_shortest_path",
    "generate_mst_prim",
    "generate_graph_traversal",
    "generate_tsp",
    "generate_convex_hull",
    "generate_delaunay",
]
