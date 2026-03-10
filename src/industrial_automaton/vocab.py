"""Neural Symbolic Language (NSL) Vocabulary.

55-token unified vocabulary across 3 semantic types.

Token Layout
============
  DIGIT      [0..34]   — values the model processes
  OPERATIONAL[35..48]  — structural/operator tokens
  SYSTEM     [49..54]  — sequence metadata tokens

DIGIT (35 tokens):
  [0..31]   D0 .. D31       base-32 digits (token ID = value)
  [32]      ZERO             structural null value (think-phase input)
  [33]      TRUE
  [34]      FALSE

OPERATIONAL (14 tokens):
  [35]  OP_ADD
  [36]  OP_SUB
  [37]  OP_MUL
  [38]  OP_EQ
  [39]  OPEN              Dyck/grouping open  — always followed by digit arg
  [40]  CLOSE             Dyck/grouping close — always followed by digit arg
  [41]  MOD               modulus prefix      — always followed by digit arg
  [42]  VAR               step output ref     — always followed by digit arg
  [43]  INPUT             Model input
  [44]  NAV_LEFT          left  / cycle-backward
  [45]  NAV_RIGHT         right / cycle-forward
  [46]  NAV_UP            up    / spatial-above
  [47]  NAV_DOWN          down  / spatial-below
  [48]  POP               a general delete replacing the STACK_POP

SYSTEM (6 tokens):
  [49]  PAD    sequence padding  — IGNORED in loss
  [50]  SEP    separator: program/args boundary AND inline separator (was HASH_SEP+SEP)
  [51]  YIELD  reading→writing boundary
  [52]  THEN   step separator in multi-step programs
  [53]  THINK  start planning phase (model receives ZERO inputs)
  [54]  TASK   task identity prefix — always followed by digit arg

NSL Sequence Format
===================
Single-step task:
  @t [tokens | INPUT] # INPUT -> OUTPUT

Multi-step program:
  @t0 [tokens | INPUT] | @t1 [tokens | x_i | INPUT] # INPUT -> OUTPUT

Think-step (planning) (during inference):
  @t [input] ? N N N ... -> OUTPUT

So a full sequence can look like this:

@t0 [tokens | INPUT] |
    @t1 [tokens | x_i | INPUT] #
    INPUT ?
    N N N ... ->
    OUTPUT

Loss Mask Rules
===============
These can be changed by the curriculum

  PAD, TASK, D_t (task arg), input tokens  → No loss (allowed if BPTT)
  THINK, ZERO (think phase)                → No loss (allowed if BPTT)
  THEN, SEP                                → No loss (allowed if BPTT)
  YIELD                                    → loss
  OUTPUT tokens                            → loss

Parameterized Token Pairs (op + digit arg)
==========================================
  OPEN  D_i   → Dyck/grouping open type i   (i in 0..15)
  CLOSE D_i   → Dyck/grouping close type i  (i in 0..15)
  MOD   D_n   → modulo by n                 (n in 1..31)
  VAR   D_i   → output of step i            (i < current step)
  TASK  D_t   → task identity t             (t in 0..31)

NSL Text Format (human-readable ↔ token mapping)
=================================================
  0..31        → D0..D31
  N            → ZERO
  T            → TRUE
  F            → FALSE
  +  -  *  =   → OP_ADD  OP_SUB  OP_MUL  OP_EQ
  [{i}  ]{i}   → OPEN D_i / CLOSE D_i        e.g. [0  ]1
  %{n}         → MOD D_n                     e.g. %5
  x{i}         → VAR D_i  (step i output)    e.g. x0  x21
  a{i}         → INPUT D_i (argument i)      e.g. a0  a3
  <            → NAV_LEFT  (left / backward)
  >            → NAV_RIGHT (right / forward)
  ^            → NAV_UP    (up / above)
  v            → NAV_DOWN  (down / below)
  $            → POP (a general delete replacing the STACK_POP)
  _            → PAD
  #            → SEP       (program/args boundary and inline separator)
  ->           → YIELD
  |            → THEN
  ?            → THINK
  @{t}         → TASK D_t                    e.g. @0  @5
"""

import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# DIGIT tokens [0..34]
# ---------------------------------------------------------------------------
ZERO  = 32   # structural null value (think-phase input)
TRUE  = 33
FALSE = 34

_MAX_DIGIT = 32  # D0..D31

def D(n: int) -> int:
    """Digit token for value n (0..31). Token ID = value."""
    assert 0 <= n < _MAX_DIGIT, f"Digit out of range: {n} (max {_MAX_DIGIT - 1})"
    return n

# ---------------------------------------------------------------------------
# OPERATIONAL tokens [35..48]
# ---------------------------------------------------------------------------
OP_ADD    = 35
OP_SUB    = 36
OP_MUL    = 37
OP_EQ     = 38
OPEN      = 39   # Dyck/grouping open  — followed by D_i
CLOSE     = 40   # Dyck/grouping close — followed by D_i
MOD       = 41   # modulus — followed by D_n
VAR       = 42   # step output ref — followed by D_i
INPUT     = 43   # model input marker (standalone, no digit arg)
NAV_LEFT  = 44   # left  / cycle-backward
NAV_RIGHT = 45   # right / cycle-forward
NAV_UP    = 46   # up    / spatial-above
NAV_DOWN  = 47   # down  / spatial-below
POP       = 48   # general delete (replaces STACK_POP)

# Aliases
STACK_POP = POP  # backward compat
NAV_FWD = NAV_RIGHT
NAV_BWD = NAV_LEFT

# ---------------------------------------------------------------------------
# SYSTEM tokens [49..54]
# ---------------------------------------------------------------------------
PAD   = 49   # sequence padding — ignored in loss
SEP   = 50   # separator: program/args boundary AND inline (was HASH_SEP + SEP)
YIELD = 51   # reading → writing boundary
THEN  = 52   # step separator
THINK = 53   # start planning phase
TASK  = 54   # task identity prefix — followed by D_t

SIZE  = 55   # total vocabulary size

# ---------------------------------------------------------------------------
# Spatial / navigation direction constants
# ---------------------------------------------------------------------------
NAV_ABOVE = NAV_UP
NAV_BELOW = NAV_DOWN
# NAV_LEFT and NAV_RIGHT serve double duty for spatial left/right

# ---------------------------------------------------------------------------
# Task registry: name → digit index for TASK D_t
# ---------------------------------------------------------------------------
TASK_NAMES = [
    "parity_check",              # D0
    "even_pairs",                # D1
    "cycle_navigation",          # D2
    "modular_arithmetic",        # D3
    "dyck_n",                    # D4
    "reverse_string",            # D5
    "stack_manipulation",        # D6
    "nested_modular_arithmetic", # D7
    "duplicate_string",          # D8
    "repeat_copy_n",             # D9
    "odds_first",                # D10
    "sort",                      # D11
    "square_root",               # D12
    "count_n",                   # D13
    "n_back",                    # D14
    "associative_recall",        # D15
    "missing_duplicate",         # D16
    "deduplicate_inputs",        # D17
    "mini_shrdlu",               # D18
    "shortest_path",             # D19
    "mst_prim",                  # D20
    "graph_traversal",           # D21
    "tsp",                       # D22
    "convex_hull",               # D23
    "delaunay",                  # D24
    # D25..D31 reserved for future tasks
]

_TASK_NAME_TO_IDX = {name: i for i, name in enumerate(TASK_NAMES)}
_MAX_TASK_SLOTS   = _MAX_DIGIT  # 32 task slots (D0..D31)


def task_idx(task_name: str) -> int:
    """Return digit index for task name (used as arg to TASK token)."""
    if task_name not in _TASK_NAME_TO_IDX:
        raise ValueError(f"Unknown task: '{task_name}'. Available: {TASK_NAMES}")
    return _TASK_NAME_TO_IDX[task_name]


def task_prefix(task_name: str) -> List[int]:
    """Return [TASK, D_i] token pair for a task name."""
    return [TASK, D(task_idx(task_name))]


# ---------------------------------------------------------------------------
# Token type classification
# ---------------------------------------------------------------------------
def token_type(t: int) -> str:
    """Return the semantic type of a token: 'digit', 'operational', or 'system'."""
    if 0 <= t <= 34:
        return "digit"
    elif 35 <= t <= 48:
        return "operational"
    elif 49 <= t <= 54:
        return "system"
    else:
        raise ValueError(f"Token {t} out of vocabulary range [0..{SIZE-1}]")


# ---------------------------------------------------------------------------
# Decode / display
# ---------------------------------------------------------------------------
_DIGIT_STR = {
    **{i: f"D{i}" for i in range(_MAX_DIGIT)},
    ZERO:  "ZERO",
    TRUE:  "TRUE",
    FALSE: "FALSE",
}

_OPERATIONAL_STR = {
    OP_ADD:    "+",
    OP_SUB:    "-",
    OP_MUL:    "*",
    OP_EQ:     "=",
    OPEN:      "<OPEN>",
    CLOSE:     "<CLOSE>",
    MOD:       "<M>",
    VAR:       "<VAR>",
    INPUT:     "INPUT",
    NAV_LEFT:  "<",
    NAV_RIGHT: ">",
    NAV_UP:    "^",
    NAV_DOWN:  "v",
    POP:       "$",
}

_SYSTEM_STR = {
    PAD:   "<PAD>",
    SEP:   "#",
    YIELD: "->",
    THEN:  "|",
    THINK: "?",
    TASK:  "@",
}

_ALL_STR = {**_DIGIT_STR, **_OPERATIONAL_STR, **_SYSTEM_STR}

# Tokens that consume the next digit as an argument (parameterized pairs)
_PAIR_HEADS = {OPEN, CLOSE, MOD, VAR, TASK}


def pretty(token_ids, *, skip_pad: bool = True) -> str:
    """Format token IDs as NSL text (parameterized pairs collapsed).

    Parameterized pairs are collapsed into single symbols:
      TASK D_t  → @t      OPEN D_i  → [i      CLOSE D_i → ]i
      MOD  D_n  → %n      VAR  D_i  → x{i}

    Single tokens use compact NSL text format:
      0-31 → digit value  N=ZERO  T=TRUE  F=FALSE
      + - * =   INPUT   < > ^ v   $   _ # -> | ? @

    Args:
        token_ids: Iterable of integer token IDs.
        skip_pad: If True (default), PAD tokens are omitted from output.

    Returns:
        Space-separated NSL text string.
    """
    tokens = [int(t) for t in token_ids]
    parts = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t == PAD:
            if not skip_pad:
                parts.append("_")
            i += 1
            continue

        if t in _PAIR_HEADS:
            # Consume next token as digit arg
            if i + 1 < len(tokens):
                arg = tokens[i + 1]
                n = arg  # digit tokens have ID == value
                if t == TASK:
                    parts.append(f"@{n}")
                elif t == OPEN:
                    parts.append(f"[{n}")
                elif t == CLOSE:
                    parts.append(f"]{n}")
                elif t == MOD:
                    parts.append(f"%{n}")
                elif t == VAR:
                    parts.append(f"x{n}")
                i += 2
            else:
                # Dangling pair head at end of sequence
                parts.append(_ALL_STR.get(t, f"<UNK:{t}>"))
                i += 1
            continue

        # Single tokens — compact NSL text
        if 0 <= t < _MAX_DIGIT:
            parts.append(str(t))
        elif t == ZERO:
            parts.append("N")
        elif t == TRUE:
            parts.append("T")
        elif t == FALSE:
            parts.append("F")
        elif t == OP_ADD:
            parts.append("+")
        elif t == OP_SUB:
            parts.append("-")
        elif t == OP_MUL:
            parts.append("*")
        elif t == OP_EQ:
            parts.append("=")
        elif t == NAV_LEFT:
            parts.append("<")
        elif t == NAV_RIGHT:
            parts.append(">")
        elif t == NAV_UP:
            parts.append("^")
        elif t == NAV_DOWN:
            parts.append("v")
        elif t == INPUT:
            parts.append("INPUT")
        elif t == POP:
            parts.append("$")
        elif t == SEP:
            parts.append("#")
        elif t == YIELD:
            parts.append("->")
        elif t == THEN:
            parts.append("|")
        elif t == THINK:
            parts.append("?")
        else:
            parts.append(f"<UNK:{t}>")

        i += 1

    return " ".join(parts)


def make_mask(token_ids) -> np.ndarray:
    """Create a boolean output mask from a set/list of valid token IDs."""
    mask = np.zeros(SIZE, dtype=bool)
    for t in token_ids:
        mask[int(t)] = True
    return mask


def describe() -> str:
    """Print a summary of the vocabulary."""
    lines = [
        f"NSL Vocabulary — {SIZE} tokens",
        f"  DIGIT      ({34 - 0 + 1} tokens): D0..D31, ZERO, TRUE, FALSE   [IDs 0..34]",
        f"  OPERATIONAL({48 - 35 + 1} tokens): operators, nav, family tokens [IDs 35..48]",
        f"  SYSTEM     ({54 - 49 + 1} tokens): PAD SEP YIELD THEN THINK TASK [IDs 49..54]",
        f"  Tasks registered: {len(TASK_NAMES)} / {_MAX_TASK_SLOTS} slots",
    ]
    return "\n".join(lines)
