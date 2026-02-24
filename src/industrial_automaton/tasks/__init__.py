from .regular import (
    generate_cycle_navigation,
    generate_even_pairs,
    generate_modular_arithmetic,
    generate_parity_check,
)
from .context_free import (
    generate_dyck_n,
    generate_nested_modular_arithmetic,
    generate_reverse_string,
    generate_solve_equation,
    generate_stack_manipulation,
)
from .context_sensitive import (
    generate_associative_recall,
    generate_count_n,
    generate_deduplicate_inputs,
    generate_duplicate_string,
    generate_missing_duplicate,
    generate_n_back,
    generate_odds_first,
    generate_repeat_copy_n,
)
from .arithmetic import (
    generate_square_root,
    generate_8_bit_addition,
    generate_16_bit_addition,
    generate_32_bit_addition,
    generate_64_bit_addition,
    generate_8_bit_multiplication,
    generate_16_bit_multiplication,
    generate_32_bit_multiplication,
)
from .data_processing import (
    generate_mini_shrdlu,
    generate_python_execution,
    generate_sort,
)
from .graphs_geometry import (
    generate_convex_hull,
    generate_delaunay,
    generate_graph_traversal,
    generate_mst_prim,
    generate_shortest_path,
    generate_tsp,
)

# Unified vocab info registry: task_name -> vocab_info_fn(**task_kwargs) -> dict
from .regular import VOCAB_INFO as _reg_vi
from .context_free import VOCAB_INFO as _cf_vi
from .context_sensitive import VOCAB_INFO as _cs_vi
from .arithmetic import VOCAB_INFO as _ar_vi
from .data_processing import VOCAB_INFO as _dp_vi
from .graphs_geometry import VOCAB_INFO as _gg_vi

VOCAB_INFO = {**_reg_vi, **_cf_vi, **_cs_vi, **_ar_vi, **_dp_vi, **_gg_vi}
