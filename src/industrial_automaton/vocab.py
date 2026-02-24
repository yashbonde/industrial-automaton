"""Unified vocabulary for all tasks.

Token layout:
  [0..99]       D0 .. D99           (digit tokens, ID = value)
  [100]         PAD
  [101]         BOS
  [102]         EOS
  [103]         SEP
  [104..139]    TASK_0 .. TASK_35   (36 task ID tokens)
  [140..143]    OP_ADD, OP_SUB, OP_MUL, OP_EQ
  [144..183]    OPEN_0..OPEN_19, CLOSE_0..CLOSE_19  (Dyck brackets)
  [184]         VAR_X
  [185]         OPEN_PAREN          (arithmetic parenthesis)
  [186]         CLOSE_PAREN
  [187]         STACK_POP
  [188]         NAV_FWD
  [189]         NAV_BWD
  [190]         HASH_SEP            (n-back separator)
  [191]         TRUE
  [192]         FALSE
  [193]         PROG_ASSIGN
  [194]         PROG_LOOP_START
  [195]         PROG_LOOP_END
  [196..199]    REL_ABOVE, REL_BELOW, REL_LEFT, REL_RIGHT
"""

import numpy as np

# -- Digit tokens: D0..D99 (token ID = digit value) --
# Access as V.D(n) or just use the integer directly.

PAD = 100
BOS = 101
EOS = 102
SEP = 103

# Task ID tokens
TASK_0 = 104
_NUM_TASK_SLOTS = 36

# Operators
OP_ADD = 140
OP_SUB = 141
OP_MUL = 142
OP_EQ = 143

# Dyck bracket tokens: OPEN_i = 144 + i, CLOSE_i = 164 + i  (i in 0..19)
_DYCK_OPEN_BASE = 144
_DYCK_CLOSE_BASE = 164
_MAX_DYCK_N = 20


def OPEN(i: int) -> int:
    """Dyck open bracket for type i (0-indexed)."""
    assert 0 <= i < _MAX_DYCK_N
    return _DYCK_OPEN_BASE + i


def CLOSE(i: int) -> int:
    """Dyck close bracket for type i (0-indexed)."""
    assert 0 <= i < _MAX_DYCK_N
    return _DYCK_CLOSE_BASE + i


VAR_X = 184
OPEN_PAREN = 185
CLOSE_PAREN = 186
STACK_POP = 187
NAV_FWD = 188
NAV_BWD = 189
HASH_SEP = 190

TRUE = 191
FALSE = 192

PROG_ASSIGN = 193
PROG_LOOP_START = 194
PROG_LOOP_END = 195

REL_ABOVE = 196
REL_BELOW = 197
REL_LEFT = 198
REL_RIGHT = 199

ZERO = 200

SIZE = 201

# ---- Task name registry (name -> index) ----
TASK_NAMES = [
    "parity_check",
    "even_pairs",
    "cycle_navigation",
    "modular_arithmetic",
    "dyck_n",
    "reverse_string",
    "stack_manipulation",
    "nested_modular_arithmetic",
    "solve_equation",
    "binary_addition",
    "binary_multiplication",
    "duplicate_string",
    "repeat_copy_n",
    "odds_first",
    "sort",
    "square_root",
    "count_n",
    "n_back",
    "associative_recall",
    "missing_duplicate",
    "deduplicate_inputs",
    "python_execution",
    "mini_shrdlu",
    "shortest_path",
    "mst_prim",
    "graph_traversal",
    "tsp",
    "convex_hull",
    "delaunay",
]

_TASK_NAME_TO_IDX = {name: i for i, name in enumerate(TASK_NAMES)}


def task_token(task_name: str) -> int:
    """Return the TASK_N token for a given task name."""
    return TASK_0 + _TASK_NAME_TO_IDX[task_name]


def D(n: int) -> int:
    """Digit token for value n. Identity function (token ID = value)."""
    assert 0 <= n < 100
    return n


def output_mask(task_name: str, **task_kwargs) -> np.ndarray:
    """Bool array of shape (SIZE,). True where output token is valid."""
    from . import tasks as _tasks
    info_fn = _tasks.VOCAB_INFO[task_name]
    info = info_fn(**task_kwargs)
    return info["output_mask"]


def decode(token_ids, *, compact: bool = False) -> list[str]:
    """Human-readable decoding of token IDs."""
    result = []
    for t in token_ids:
        t = int(t)
        if 0 <= t <= 99:
            result.append(f"D{t}" if not compact else str(t))
        elif t == PAD:
            result.append("<PAD>")
        elif t == BOS:
            result.append("<BOS>")
        elif t == EOS:
            result.append("<EOS>")
        elif t == SEP:
            result.append("<SEP>")
        elif TASK_0 <= t < TASK_0 + _NUM_TASK_SLOTS:
            idx = t - TASK_0
            name = TASK_NAMES[idx] if idx < len(TASK_NAMES) else f"task_{idx}"
            result.append(f"<TASK:{name}>")
        elif t == OP_ADD:
            result.append("+")
        elif t == OP_SUB:
            result.append("-")
        elif t == OP_MUL:
            result.append("*")
        elif t == OP_EQ:
            result.append("=")
        elif _DYCK_OPEN_BASE <= t < _DYCK_CLOSE_BASE:
            result.append(f"OPEN_{t - _DYCK_OPEN_BASE}")
        elif _DYCK_CLOSE_BASE <= t < _DYCK_CLOSE_BASE + _MAX_DYCK_N:
            result.append(f"CLOSE_{t - _DYCK_CLOSE_BASE}")
        elif t == VAR_X:
            result.append("x")
        elif t == OPEN_PAREN:
            result.append("(")
        elif t == CLOSE_PAREN:
            result.append(")")
        elif t == STACK_POP:
            result.append("POP")
        elif t == NAV_FWD:
            result.append("FWD")
        elif t == NAV_BWD:
            result.append("BWD")
        elif t == HASH_SEP:
            result.append("#")
        elif t == TRUE:
            result.append("TRUE")
        elif t == FALSE:
            result.append("FALSE")
        elif t == PROG_ASSIGN:
            result.append("ASSIGN")
        elif t == PROG_LOOP_START:
            result.append("LOOP_START")
        elif t == PROG_LOOP_END:
            result.append("LOOP_END")
        elif t == REL_ABOVE:
            result.append("ABOVE")
        elif t == REL_BELOW:
            result.append("BELOW")
        elif t == REL_LEFT:
            result.append("LEFT")
        elif t == REL_RIGHT:
            result.append("RIGHT")
        elif t == ZERO:
            result.append("<ZERO>")
        else:
            result.append(f"<UNK:{t}>")
    return result


def _make_mask(token_ids) -> np.ndarray:
    """Create a boolean output mask from a set/list of valid token IDs."""
    mask = np.zeros(SIZE, dtype=bool)
    for t in token_ids:
        mask[t] = True
    return mask
