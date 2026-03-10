"""Unified task registry with decorator-based auto-registration.

All task generators are registered here with metadata including:
- Parameter presets (baseline, max, in_papercode)
- Task constraints (min_length)

This file contains all 29 tasks in a single, greppable location.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set
import numpy as np

from .. import vocab as V

# ============================================================================
# Constants and Helpers (shared across all tasks)
# ============================================================================

_MAX_SEQLEN = 10_000
_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _default_rng(rng):
    """Return a default RNG if None is provided."""
    if rng is None:
        return np.random.default_rng()
    return rng


def _check_seqlen(total, limit=_MAX_SEQLEN, **ctx):
    """Validate that sequence length doesn't exceed limit."""
    if total > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(
            f"total seqlen {total} exceeds limit {limit} ({detail})")


def _strip_pad(arr):
    """Convert numpy row to list, stripping trailing PAD tokens."""
    lst = [int(x) for x in arr]
    while lst and lst[-1] == V.PAD:
        lst.pop()
    return lst


# ============================================================================
# Registry Data Structures
# ============================================================================

@dataclass(frozen=True)
class TaskEntry:
    """Metadata for a registered task. LSP autocompletes all fields."""
    fn: Callable                        # The generator function
    category: str                       # "regular", "context_free", etc.
    baseline: Dict[str, Any]            # {"length": 10, "vocab_size": 2}
    max: Dict[str, Any]                 # {"length": 1000, "vocab_size": 10}
    in_papercode: Dict[str, Any]        # {"length": 50, "vocab_size": 2}
    min_length: Optional[int] = None    # Hard minimum
    output_vocab: Optional[np.ndarray] = None  # (V.SIZE,) bool mask of valid output tokens


class Registry(dict):
    """Custom registry supporting both dict-style and dot-style access.

    Usage:
        MASTER_REGISTRY["even_pairs"]       # Dict-style
        MASTER_REGISTRY.even_pairs          # Dot-style (LSP autocomplete)
    """

    def __getattr__(self, name: str) -> TaskEntry:
        """Enable dot-style access for LSP autocomplete."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Task '{name}' not found in registry")

    def __setattr__(self, name: str, value: TaskEntry):
        """Enable dot-style assignment."""
        self[name] = value

    def __repr__(self) -> str:
        """Pretty-print registered tasks."""
        tasks_by_category = {}
        for name, entry in self.items():
            cat = entry.category
            if cat not in tasks_by_category:
                tasks_by_category[cat] = []
            tasks_by_category[cat].append(name)

        lines = ["Task Registry:"]
        for cat, tasks in sorted(tasks_by_category.items()):
            lines.append(f"  {cat}: {', '.join(sorted(tasks))}")
        return "\n".join(lines)


# Global registry instance
MASTER_REGISTRY = Registry()


# ============================================================================
# Task Decorator
# ============================================================================

def task(
    category: str,
    baseline: Dict[str, Any],
    max: Dict[str, Any],
    in_papercode: Dict[str, Any],
    min_length: Optional[int] = None,
    output_vocab: Optional[Set[int]] = None,
):
    """Decorator to register a task generator.

    Args:
        category: Task category ("regular", "context_free", etc.)
        baseline: Default parameters for normal training
        max: Stress-test parameters
        in_papercode: Paper reproduction parameters
        min_length: Minimum sequence length allowed

    Example:
        @task(
            category="regular",
            baseline={"length": 10, "vocab_size": 2},
            max={"length": 1000, "vocab_size": 10},
            in_papercode={"length": 50, "vocab_size": 2},
        )
        def generate_even_pairs(rng, batch_size, length, *, vocab_size=2):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        # Create task entry
        entry = TaskEntry(
            fn=fn,
            category=category,
            baseline=baseline,
            max=max,
            in_papercode=in_papercode,
            min_length=min_length,
            output_vocab=V.make_mask(output_vocab) if output_vocab is not None else None,
        )

        # Auto-register in global registry
        task_name = fn.__name__.replace("generate_", "")
        MASTER_REGISTRY[task_name] = entry

        # Return original function unchanged
        return fn

    return decorator


# ============================================================================
# Task Generators - Regular Tasks (solvable by finite automata)
# ============================================================================

@task(
    category="regular",
    baseline={"length": 10, "vocab_size": 2},
    max={"length": 1000, "vocab_size": 10},
    in_papercode={"length": 50, "vocab_size": 2},
    output_vocab={V.TRUE, V.FALSE},
)
def generate_even_pairs(rng, batch_size, length, *, vocab_size=2):
    """Check if symbols appear in identical adjacent pairs.

    Vocab: digits D1..D(vocab_size). Label TRUE/FALSE.
    """
    rng = _default_rng(rng)
    _check_seqlen(length + 1, length=length)
    rng = _default_rng(rng)
    # Tokens are D1..D(vocab_size) — i.e. integer values 1..vocab_size
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    labels_bool = np.ones(batch_size, dtype=bool)
    for i in range(0, length - 1, 2):
        labels_bool &= (seqs[:, i] == seqs[:, i + 1])
    if length % 2 == 1:
        labels_bool[:] = False

    labels = np.where(labels_bool, V.TRUE, V.FALSE).astype(np.int64)

    in_fmt = [" ".join(_ALPHA[t - 1] for t in seqs[b]) for b in range(batch_size)]
    out_fmt = ["yes" if labels_bool[b] else "no" for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [[int(labels[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="regular",
    baseline={"length": 10, "vocab_size": 2},
    max={"length": 1000, "vocab_size": 10},
    in_papercode={"length": 50, "vocab_size": 2},
    output_vocab={V.TRUE, V.FALSE},
)
def generate_parity_check(rng, batch_size, length, *, symbol=1, vocab_size=2):
    """Determine if the count of `symbol` in the sequence is even or odd.

    Output: TRUE (odd count) or FALSE (even count).
    """
    rng = _default_rng(rng)
    _check_seqlen(length + 1, length=length)
    rng = _default_rng(rng)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    counts = np.sum(seqs == symbol, axis=1)
    is_odd = (counts % 2).astype(bool)
    labels = np.where(is_odd, V.TRUE, V.FALSE).astype(np.int64)

    sym_char = _ALPHA[symbol - 1]
    in_fmt = [" ".join(_ALPHA[t - 1] for t in seqs[b]) for b in range(batch_size)]
    out_fmt = [f"odd (count({sym_char})={int(counts[b])})" if is_odd[b]
               else f"even (count({sym_char})={int(counts[b])})" for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [[int(labels[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="regular",
    baseline={"length": 10, "num_states": 5},
    max={"length": 5000, "num_states": 31},
    in_papercode={"length": 50, "num_states": 5},
    output_vocab=set(range(32)),  # cycle position = D0..D31
)
def generate_cycle_navigation(rng, batch_size, length, *, num_states=5):
    """Navigate a cycle of `num_states` states. NAV_FWD=forward, NAV_BWD=backward.

    Output is the final state as D(state).
    """
    rng = _default_rng(rng)
    _check_seqlen(length + 1, length=length)
    rng = _default_rng(rng)
    # Choose FWD or BWD
    choices = rng.integers(0, 2, size=(batch_size, length))
    seqs = np.where(choices == 0, V.NAV_RIGHT, V.NAV_LEFT).astype(np.int64)

    pos = np.zeros(batch_size, dtype=np.int64)
    for t in range(length):
        step = np.where(seqs[:, t] == V.NAV_RIGHT, 1, -1)
        pos = (pos + step) % num_states

    arrow = {V.NAV_RIGHT: "->", V.NAV_LEFT: "<-"}
    in_fmt = [" ".join(arrow[int(seqs[b, t])] for t in range(length)) for b in range(batch_size)]
    out_fmt = [f"state {pos[b]}" for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [[int(pos[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="regular",
    baseline={"length": 9, "modulus": 5},
    max={"length": 999, "modulus": 31},
    in_papercode={"length": 51, "modulus": 5},
    output_vocab=set(range(32)),  # result = D0..D31
)
def generate_modular_arithmetic(rng, batch_size, length, *, modulus=5):
    """Evaluate simple arithmetic expressions (no brackets) under modulus.

    Encoding: operands D0..D(modulus-1), operators OP_ADD/OP_SUB/OP_MUL.
    Output: D(result).
    """
    rng = _default_rng(rng)
    # Adjust to nearest odd length (required for operand/operator alternation)
    if length % 2 == 0:
        length = length - 1 if length > 1 else 1
    _check_seqlen(length + 1, length=length)

    num_operands = (length + 1) // 2
    num_operators = length // 2

    operands = rng.integers(0, modulus, size=(batch_size, num_operands))
    operators = rng.integers(0, 3, size=(batch_size, num_operators))  # 0=+, 1=-, 2=*

    # Build encoded sequences — operands are D(value), operators are OP tokens
    seqs = np.zeros((batch_size, length), dtype=np.int64)
    for i in range(num_operands):
        seqs[:, 2 * i] = operands[:, i]  # D(value) = value itself
    op_tokens = np.array([V.OP_ADD, V.OP_SUB, V.OP_MUL])
    for i in range(num_operators):
        seqs[:, 2 * i + 1] = op_tokens[operators[:, i]]

    # Evaluate left-to-right (no operator precedence)
    result = operands[:, 0] % modulus
    for i in range(num_operators):
        op = operators[:, i]
        val = operands[:, i + 1]
        add_mask = op == 0
        sub_mask = op == 1
        mul_mask = op == 2
        result = np.where(add_mask, (result + val) % modulus, result)
        result = np.where(sub_mask, (result - val) % modulus, result)
        result = np.where(mul_mask, (result * val) % modulus, result)

    op_sym = {0: "+", 1: "-", 2: "*"}
    in_fmt = []
    for b in range(batch_size):
        parts = [str(operands[b, 0])]
        for i in range(num_operators):
            parts.append(op_sym[int(operators[b, i])])
            parts.append(str(operands[b, i + 1]))
        in_fmt.append(f"({' '.join(parts)}) mod {modulus}")
    out_fmt = [str(int(result[b])) for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [[int(result[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# ============================================================================
# Context-Free Tasks (require stack memory)
# ============================================================================

_MAX_DYCK_N = 20  # capped by vocab layout (20 open + 20 close tokens)
_BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


@task(
    category="context_free",
    baseline={"length": 10, "vocab_size": 4},
    max={"length": 1000, "vocab_size": 10},
    in_papercode={"length": 50, "vocab_size": 4},
    output_vocab=set(range(1, 11)) | {V.ZERO},  # D1..D10 + ZERO (empty stack)
)
def generate_stack_manipulation(rng, batch_size: int, length: int, *, vocab_size: int = 4):
    """Execute push/pop instructions, output top-of-stack after each step.

    Encoding: D1..D(vocab_size) = push that value, STACK_POP = pop.
    Output: top-of-stack value as D(val) after each instruction (V.PAD if stack empty).
    Shapes: input (batch_size, length), output (batch_size, length).
    """
    rng = _default_rng(rng)
    _check_seqlen(2 * length, length=length)
    rng = _default_rng(rng)
    # Generate: values 1..vocab_size or STACK_POP
    # We'll generate random choices first
    is_pop = rng.random(size=(batch_size, length)) < (1.0 / (vocab_size + 1))
    vals = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    seqs = np.where(is_pop, V.STACK_POP, vals).astype(np.int64)

    outputs = np.full((batch_size, length), V.PAD, dtype=np.int64)

    for b in range(batch_size):
        stack = []
        for t in range(length):
            token = seqs[b, t]
            if token == V.STACK_POP:
                if stack:
                    stack.pop()
            else:
                stack.append(int(token))
            outputs[b, t] = stack[-1] if stack else V.PAD

    in_fmt = []
    out_fmt = []
    for b in range(batch_size):
        ops = []
        for t in range(length):
            tok = int(seqs[b, t])
            if tok == V.STACK_POP:
                ops.append("pop")
            else:
                ops.append(f"push({tok})")
        in_fmt.append(" ".join(ops))
        tops = [str(int(outputs[b, t])) if outputs[b, t] != V.PAD else "_" for t in range(length)]
        out_fmt.append("top: [" + " ".join(tops) + "]")

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_free",
    baseline={"length": 10, "vocab_size": 8},
    max={"length": 5000, "vocab_size": 26},
    in_papercode={"length": 50, "vocab_size": 8},
    output_vocab=set(range(1, 27)),  # D1..D26
)
def generate_reverse_string(rng, batch_size, length, *, vocab_size: int = 8):
    """Recall the input sequence in reverse order.

    Input: D1..D(vocab_size). Output: same tokens reversed.
    """
    rng = _default_rng(rng)
    _check_seqlen(2 * length, length=length)
    rng = _default_rng(rng)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    outputs = seqs[:, ::-1].copy()

    alpha = "abcdefghijklmnopqrstuvwxyz"
    in_fmt = [" ".join(alpha[t - 1] for t in seqs[b]) for b in range(batch_size)]
    out_fmt = [" ".join(alpha[t - 1] for t in outputs[b]) for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def _gen_expr(rng, modulus, max_depth, depth):
    """Recursively generate a modular arithmetic expression using unified tokens."""
    if depth >= max_depth or (depth > 0 and rng.random() < 0.4):
        val = rng.integers(0, modulus)
        return [int(val)], int(val)  # D(val) = val

    left_tokens, left_val = _gen_expr(rng, modulus, max_depth, depth + 1)
    right_tokens, right_val = _gen_expr(rng, modulus, max_depth, depth + 1)
    op_choice = rng.integers(0, 3)
    op_token = [V.OP_ADD, V.OP_SUB, V.OP_MUL][op_choice]

    if op_choice == 0:
        result = (left_val + right_val) % modulus
    elif op_choice == 1:
        result = (left_val - right_val) % modulus
    else:
        result = (left_val * right_val) % modulus

    if depth > 0:
        tokens = [V.OPEN, 0] + left_tokens + [op_token] + right_tokens + [V.CLOSE, 0]
    else:
        tokens = left_tokens + [op_token] + right_tokens

    return tokens, result


@task(
    category="context_free",
    baseline={"length": 10, "modulus": 5, "max_depth": 3},
    max={"length": 1000, "modulus": 10, "max_depth": 5},
    in_papercode={"length": 50, "modulus": 5, "max_depth": 3},
    output_vocab=set(range(32)),  # result mod N = D0..D31
)
def generate_nested_modular_arithmetic(rng, batch_size: int, length: int, *, modulus: int=5, max_depth: int=3):
    """Modular arithmetic with nested parentheses.

    Encoding: operands D0..D(modulus-1), operators OP_ADD/OP_SUB/OP_MUL,
    OPEN D0 / CLOSE D0 for grouping. Output: D(result).
    Shapes: input (batch_size, length) padded with PAD, output (batch_size,).
    """
    rng = _default_rng(rng)
    _check_seqlen(length + 1, length=length)
    rng = _default_rng(rng)

    all_seqs = []
    all_results = []

    for _ in range(batch_size):
        tokens, val = _gen_expr(rng, modulus, max_depth, 0)
        all_seqs.append(tokens)
        all_results.append(val % modulus)

    max_len = max(len(s) for s in all_seqs)
    padded_len = max(max_len, length)
    if padded_len + 1 > _MAX_SEQLEN:
        padded_len = _MAX_SEQLEN - 1
    inputs = np.full((batch_size, padded_len), V.PAD, dtype=np.int64)
    for i, s in enumerate(all_seqs):
        end = min(len(s), padded_len)
        inputs[i, :end] = s[:end]

    tok_map = {V.OP_ADD: "+", V.OP_SUB: "-", V.OP_MUL: "*",
               V.OPEN: "[0", V.CLOSE: "]0"}
    in_fmt = []
    for b in range(batch_size):
        parts = []
        for tok in all_seqs[b]:
            if tok in tok_map:
                parts.append(tok_map[tok])
            else:
                parts.append(str(tok))  # D(val) = val itself
        in_fmt.append(f"({''.join(parts)}) mod {modulus}")
    out_fmt = [str(int(all_results[b])) for b in range(batch_size)]

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [[int(all_results[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# NOTE: solve_equation commented out — required VAR_X (algebraic x) which is removed from NSL
# @task(
#     category="context_free",
#     baseline={"length": 7, "max_val": 9},
#     max={"length": 7, "max_val": 31},
#     in_papercode={"length": 7, "max_val": 9},
#     output_vocab=set(range(32)) | {V.OP_ADD, V.OP_SUB, V.OP_MUL},
# )
def generate_solve_equation(rng, batch_size, length, *, max_val=9):
    """Find the value of x in a simple linear equation: a * x + b = c.

    Encoding: D(a), OP_MUL, VAR_X, OP_ADD, D(b), OP_EQ, D(c).
    Output: D(x).
    """
    rng = _default_rng(rng)
    actual_in = max(length, 7)
    _check_seqlen(actual_in + 1, length=length)
    rng = _default_rng(rng)

    inputs = np.full((batch_size, actual_in), V.PAD, dtype=np.int64)
    outputs = np.zeros(batch_size, dtype=np.int64)

    in_fmt = []
    out_fmt = []
    for b in range(batch_size):
        x = rng.integers(0, max_val + 1)
        a = rng.integers(1, max_val + 1)  # nonzero
        bias = rng.integers(0, max_val + 1)
        c = (a * x + bias) % (max_val + 1)

        inputs[b, 0] = int(a)      # D(a)
        inputs[b, 1] = V.OP_MUL
        inputs[b, 2] = V.VAR_X
        inputs[b, 3] = V.OP_ADD
        inputs[b, 4] = int(bias)   # D(bias)
        inputs[b, 5] = V.OP_EQ
        inputs[b, 6] = int(c)      # D(c)
        outputs[b] = int(x)        # D(x)

        in_fmt.append(f"{a} * x + {bias} = {c}")
        out_fmt.append(f"x = {int(x)}")

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [[int(outputs[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def _gen_dyck_balanced(rng, n, num_pairs):
    """Generate a balanced Dyck-n token list with num_pairs bracket pairs.

    Each bracket is a two-token pair: [OPEN D_i] or [CLOSE D_i].
    Returns a flat list of tokens with length == 2 * num_pairs (or less if padded).
    """
    tokens = []
    stack = []
    pairs_remaining = num_pairs
    while pairs_remaining > 0:
        open_needed = len(stack)  # must close these before we're done
        if open_needed == pairs_remaining:
            # Must close everything now
            bracket_type = stack.pop()
            tokens.extend([V.CLOSE, bracket_type])
            pairs_remaining -= 1
        elif stack and rng.random() < 0.4:
            bracket_type = stack.pop()
            tokens.extend([V.CLOSE, bracket_type])
            pairs_remaining -= 1
        else:
            bracket_type = int(rng.integers(0, n))
            stack.append(bracket_type)
            tokens.extend([V.OPEN, bracket_type])
            pairs_remaining -= 1
    return tokens


def _check_dyck_balanced(flat_tokens):
    """Check if a flat two-token-pair sequence is a valid Dyck word.

    Expects [OPEN D_i] / [CLOSE D_i] pairs; skips PAD tokens.
    """
    toks = [int(t) for t in flat_tokens if int(t) != V.PAD]
    stack = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == V.OPEN:
            if i + 1 >= len(toks):
                return False
            stack.append(toks[i + 1])
            i += 2
        elif tok == V.CLOSE:
            if i + 1 >= len(toks):
                return False
            bracket_type = toks[i + 1]
            if not stack or stack[-1] != bracket_type:
                return False
            stack.pop()
            i += 2
        else:
            i += 1  # unexpected token — skip
    return len(stack) == 0


@task(
    category="context_free",
    baseline={"length": 10, "n": 2},
    max={"length": 1000, "n": 15},
    in_papercode={"length": 50, "n": 2},
    output_vocab={V.TRUE, V.FALSE},
)
def generate_dyck_n(rng, batch_size, length, *, n: int = 2):
    """Validate balanced parentheses with n bracket types.

    Each bracket is a two-token NSL pair: <OPEN> D_i or <CLOSE> D_i.
    `length` = number of bracket pairs; token sequence length = 2*length.
    Output: TRUE if balanced, FALSE otherwise.
    """
    rng = _default_rng(rng)
    if n > _MAX_DYCK_N:
        raise ValueError(f"n={n} exceeds max {_MAX_DYCK_N}")

    all_seqs = []
    labels = []

    for b in range(batch_size):
        if rng.random() < 0.5:
            tokens = _gen_dyck_balanced(rng, n, length)
            is_balanced = len(tokens) == 2 * length
        else:
            # Random bracket sequence (may or may not be balanced)
            tokens = []
            for _ in range(length):
                bracket_idx = int(rng.integers(0, 2 * n))
                if bracket_idx < n:
                    tokens.extend([V.OPEN, bracket_idx])
                else:
                    tokens.extend([V.CLOSE, bracket_idx - n])
            is_balanced = _check_dyck_balanced(tokens)

        all_seqs.append(tokens)
        labels.append(V.TRUE if is_balanced else V.FALSE)

    # Formatting
    if n <= len(_BRACKET_PAIRS):
        brackets = _BRACKET_PAIRS[:n]
        in_fmt = []
        for b in range(batch_size):
            chars = []
            toks = all_seqs[b]
            i = 0
            while i < len(toks):
                if toks[i] == V.OPEN and i + 1 < len(toks):
                    bi = toks[i + 1]
                    chars.append(brackets[bi][0] if bi < n else f"o{bi}")
                    i += 2
                elif toks[i] == V.CLOSE and i + 1 < len(toks):
                    bi = toks[i + 1]
                    chars.append(brackets[bi][1] if bi < n else f"c{bi}")
                    i += 2
                else:
                    i += 1
            in_fmt.append("".join(chars))
    else:
        in_fmt = []
        for b in range(batch_size):
            parts = []
            toks = all_seqs[b]
            i = 0
            while i < len(toks):
                if toks[i] == V.OPEN and i + 1 < len(toks):
                    parts.append(f"o{toks[i+1]}")
                    i += 2
                elif toks[i] == V.CLOSE and i + 1 < len(toks):
                    parts.append(f"c{toks[i+1]}")
                    i += 2
                else:
                    i += 1
            in_fmt.append(" ".join(parts))

    out_fmt = ["balanced" if lbl == V.TRUE else "unbalanced" for lbl in labels]

    return {"input": all_seqs,
            "output": [[lbl] for lbl in labels],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# ============================================================================
# Context-Sensitive Tasks (require tape/matrix memory)
# ============================================================================

_MAX_COUNT_N = 500
_MAX_REPEAT_COPY_LEN = 2_000
_MAX_REPEAT_COPY_N = 50


def _seq_to_letters(seq):
    return " ".join(_ALPHA[t - 1] for t in seq if 1 <= t <= 26)


@task(
    category="context_sensitive",
    baseline={"length": 10, "vocab_size": 8},
    max={"length": 500, "vocab_size": 26},
    in_papercode={"length": 50, "vocab_size": 8},
    output_vocab=set(range(1, 27)),  # D1..D26
)
def generate_duplicate_string(rng, batch_size, length, *, vocab_size=8):
    """Duplicate input: w -> ww.

    Input: (batch_size, length) with D1..D(vocab_size).
    Output: (batch_size, 2*length).
    """
    rng = _default_rng(rng)
    _check_seqlen(3 * length, length=length)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    outputs = np.concatenate([seqs, seqs], axis=1)

    in_fmt = [_seq_to_letters(seqs[b]) for b in range(batch_size)]
    out_fmt = [_seq_to_letters(outputs[b]) for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_sensitive",
    baseline={"length": 10, "max_n": 5, "vocab_size": 8},
    max={"length": 100, "max_n": 50, "vocab_size": 26},
    in_papercode={"length": 20, "max_n": 5, "vocab_size": 8},
    output_vocab=set(range(1, 27)),  # D1..D26
)
def generate_repeat_copy_n(rng, batch_size, length, *, max_n=5, vocab_size=8):
    """Reproduce an input string N times.

    Input: [D(n), tok1, tok2, ...] where n is repeat count.
    Output: pattern repeated n times, PAD-padded.
    """
    rng = _default_rng(rng)
    if length < 2:
        raise ValueError("length must be >= 2 (1 for n + at least 1 token)")
    if length - 1 > _MAX_REPEAT_COPY_LEN:
        raise ValueError(
            f"pattern length={length - 1} exceeds max {_MAX_REPEAT_COPY_LEN}")
    if max_n > _MAX_REPEAT_COPY_N:
        raise ValueError(f"max_n={max_n} exceeds max {_MAX_REPEAT_COPY_N}")
    pattern_len = length - 1
    _check_seqlen(length + max_n * pattern_len,
                  length=length, max_n=max_n)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    out_len = max_n * pattern_len
    outputs = np.full((batch_size, out_len), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []
    for b in range(batch_size):
        n = int(rng.integers(1, max_n + 1))
        pattern = rng.integers(1, vocab_size + 1, size=pattern_len)
        inputs[b, 0] = n  # D(n) — n is the repeat count
        inputs[b, 1:] = pattern
        repeated = np.tile(pattern, n)
        outputs[b, :len(repeated)] = repeated

        in_fmt.append(f"x{n} {_seq_to_letters(pattern)}")
        out_fmt.append(_seq_to_letters(repeated))

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [_strip_pad(outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_sensitive",
    baseline={"length": 10, "repeat": 3, "vocab_size": 8},
    max={"length": 1000, "repeat": 10, "vocab_size": 26},
    in_papercode={"length": 20, "repeat": 3, "vocab_size": 8},
)
def generate_deduplicate_inputs(rng, batch_size, length, *, repeat: int = 3, vocab_size: int = 8):
    """Filter redundant stream: each symbol repeated `repeat` times -> unique symbols.

    Input: (batch_size, length * repeat) with D1..D(vocab_size).
    Output: (batch_size, length).
    """
    rng = _default_rng(rng)
    input_len = length * repeat
    _check_seqlen(input_len, limit=_MAX_SEQLEN,
                  length=length, repeat=repeat, note="input seqlen")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2 to avoid adjacent duplicates")
    unique = np.zeros((batch_size, length), dtype=np.int64)
    unique[:, 0] = rng.integers(1, vocab_size + 1, size=batch_size)
    for t in range(1, length):
        candidates = rng.integers(1, vocab_size, size=batch_size)
        candidates = np.where(candidates >= unique[:, t - 1],
                              candidates + 1, candidates)
        unique[:, t] = candidates
    expanded = np.repeat(unique, repeat, axis=1)

    in_fmt = [_seq_to_letters(expanded[b]) for b in range(batch_size)]
    out_fmt = [_seq_to_letters(unique[b]) for b in range(batch_size)]

    return {"input": [list(int(x) for x in expanded[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in unique[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_sensitive",
    baseline={"length": 20, "num_pairs": 5, "vocab_size": 8},
    max={"length": 1000, "num_pairs": 100, "vocab_size": 26},
    in_papercode={"length": 50, "num_pairs": 10, "vocab_size": 8},
)
def generate_associative_recall(rng, batch_size, length, *, num_pairs=None, vocab_size=8):
    """Retrieve value for a queried key from key-value pairs.

    Input layout: [k1, v1, k2, v2, ..., kN, vN, query_key, PAD].
    Output: D(value).
    """
    rng = _default_rng(rng)
    _check_seqlen(length + 1, length=length)
    if num_pairs is None:
        num_pairs = max(1, (length - 2) // 2)
    num_pairs = min(num_pairs, vocab_size) # Ensure num_pairs <= vocab_size

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.zeros(batch_size, dtype=np.int64)

    in_fmt = []
    out_fmt = []
    for b in range(batch_size):
        keys = rng.choice(np.arange(1, vocab_size + 1), size=num_pairs, replace=False)
        values = rng.integers(1, vocab_size + 1, size=num_pairs)
        query_idx = rng.integers(0, num_pairs)

        pos = 0
        for i in range(num_pairs):
            if pos + 1 < length:
                inputs[b, pos] = keys[i]
                inputs[b, pos + 1] = values[i]
                pos += 2
        if pos < length:
            inputs[b, pos] = keys[query_idx]

        outputs[b] = values[query_idx]

        pairs_str = ", ".join(f"{_ALPHA[k-1]}:{_ALPHA[v-1]}" for k, v in zip(keys, values))
        query_char = _ALPHA[keys[query_idx] - 1]
        in_fmt.append(f"{{{pairs_str}}} ? {query_char}")
        out_fmt.append(_ALPHA[int(values[query_idx]) - 1])

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [[int(outputs[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_sensitive",
    baseline={"length": 10, "vocab_size": 12},
    max={"length": 100, "vocab_size": 102},
    in_papercode={"length": 20, "vocab_size": 22},
)
def generate_missing_duplicate(rng, batch_size, length, *, vocab_size=None):
    """Identify the missing element from a duplicated sequence.

    Input: (batch_size, 2*length - 1) with D1..D(vocab_size).
    Output: D(missing_element).
    """
    rng = _default_rng(rng)
    _check_seqlen(2 * length, length=length)
    if vocab_size is None:
        vocab_size = length + 2

    inputs_list = []
    outputs = np.zeros(batch_size, dtype=np.int64)

    for b in range(batch_size):
        elements = rng.choice(np.arange(1, vocab_size + 1), size=length, replace=False)
        missing_idx = rng.integers(0, length)
        outputs[b] = elements[missing_idx]

        doubled = []
        for i in range(length):
            doubled.append(elements[i])
            if i != missing_idx:
                doubled.append(elements[i])
        rng.shuffle(doubled)
        inputs_list.append(doubled)

    seq_len = 2 * length - 1
    inputs = np.full((batch_size, seq_len), V.PAD, dtype=np.int64)
    for b, seq in enumerate(inputs_list):
        inputs[b, :len(seq)] = seq

    in_fmt = [" ".join(str(int(t)) for t in inputs[b] if t != V.PAD) for b in range(batch_size)]
    out_fmt = [f"missing: {int(outputs[b])}" for b in range(batch_size)]

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [[int(outputs[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_sensitive",
    baseline={"length": 10, "vocab_size": 10},
    max={"length": 1000, "vocab_size": 100},
    in_papercode={"length": 50, "vocab_size": 10},
)
def generate_odds_first(rng, batch_size, length, *, vocab_size=10):
    """Reorder so odd-valued elements come first, then even-valued.

    Input/Output: D1..D(vocab_size).
    """
    rng = _default_rng(rng)
    _check_seqlen(2 * length, length=length)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    outputs = np.zeros_like(seqs)

    for b in range(batch_size):
        odd_vals = seqs[b, seqs[b] % 2 == 1]
        even_vals = seqs[b, seqs[b] % 2 == 0]
        outputs[b] = np.concatenate([odd_vals, even_vals])

    in_fmt = [" ".join(str(int(t)) for t in seqs[b]) for b in range(batch_size)]
    out_fmt = [" ".join(str(int(t)) for t in outputs[b]) for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def _check_count_n_valid(seq, n, length):
    """Check if seq is a valid s1^k s2^k ... sn^k sequence."""
    if length % n != 0:
        return False
    k = length // n
    if k == 0:
        return False
    for sym in range(n):
        block = seq[sym * k:(sym + 1) * k]
        if not np.all(block == sym + 1):
            return False
    return True


@task(
    category="context_sensitive",
    baseline={"length": 15, "n": 3},
    max={"length": 1000, "n": 100},
    in_papercode={"length": 50, "n": 5},
    output_vocab={V.TRUE, V.FALSE},
)
def generate_count_n(rng, batch_size, length, *, n: int = 3):
    """Validate s1^k s2^k ... sn^k: n symbol types each appearing k times.

    Encoding: D1..D(n). Output: TRUE/FALSE.
    """
    rng = _default_rng(rng)
    if n > _MAX_COUNT_N:
        raise ValueError(f"n={n} exceeds max {_MAX_COUNT_N}")
    _check_seqlen(length + 1, length=length, n=n)
    seqs = np.zeros((batch_size, length), dtype=np.int64)
    labels = np.full(batch_size, V.FALSE, dtype=np.int64)

    for b in range(batch_size):
        if rng.random() < 0.5 and length % n == 0:
            k = length // n
            for sym in range(n):
                seqs[b, sym * k:(sym + 1) * k] = sym + 1  # D(sym+1)
            labels[b] = V.TRUE
        else:
            seqs[b] = rng.integers(1, n + 1, size=length)
            labels[b] = V.TRUE if _check_count_n_valid(seqs[b], n, length) else V.FALSE

    if n <= 26:
        in_fmt = ["".join(_ALPHA[int(seqs[b, t]) - 1] for t in range(length))
                  for b in range(batch_size)]
    else:
        in_fmt = [" ".join(f"s{int(seqs[b, t])}" for t in range(length))
                  for b in range(batch_size)]
    out_fmt = ["valid" if labels[b] == V.TRUE else "invalid" for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [[int(labels[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


@task(
    category="context_sensitive",
    baseline={"length": 10, "n": None, "vocab_size": 8},
    max={"length": 1000, "n": None, "vocab_size": 26},
    in_papercode={"length": 50, "n": None, "vocab_size": 8},
    output_vocab=set(range(1, 27)),  # symbols D1..D26
)
def generate_n_back(rng, batch_size, length, *, n=None, vocab_size=8):
    """N-back temporal recall task.

    Input layout: [s1, s2, ..., sL, HASH_SEP, D(n)].
    Output: D(symbol) — the symbol n positions before the last.
    """
    rng = _default_rng(rng)
    _check_seqlen(2 * length + 2, length=length)

    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length),
                        dtype=np.int64)

    if n is not None:
        if n < 1 or n >= length:
            raise ValueError(f"n must be in [1, {length - 1}], got {n}")
        ns = np.full(batch_size, n, dtype=np.int64)
    else:
        ns = rng.integers(1, length, size=batch_size, dtype=np.int64)

    inputs = np.full((batch_size, length + 2), V.PAD, dtype=np.int64)
    inputs[:, :length] = seqs
    inputs[:, length] = V.SEP
    inputs[:, length + 1] = ns  # D(n)

    outputs = np.array([seqs[b, length - 1 - ns[b]] for b in range(batch_size)],
                       dtype=np.int64)

    in_fmt = [_seq_to_letters(seqs[b]) + f" # {ns[b]}"
              for b in range(batch_size)]
    out_fmt = [_ALPHA[outputs[b] - 1] for b in range(batch_size)]

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [[int(outputs[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# ============================================================================
# Arithmetic Tasks
# ============================================================================

import math

_MAX_BINARY_ADD_BITS = 64
_MAX_BINARY_MUL_BITS = 32


def _max_digits_64(base):
    """Max number of digits so that a base-`base` number fits in 64 bits."""
    return int(64 * math.log(2) / math.log(base))


def _bits_to_binstr(bits):
    """Convert LSB-first bit array to human-readable MSB-first binary string."""
    s = "".join(str(int(b)) for b in reversed(bits))
    return s.lstrip("0") or "0"


def _bits_to_int(bits):
    return sum(int(bits[i]) << i for i in range(len(bits)))


@task(
    category="arithmetic",
    baseline={"length": 5, "base": 10},
    max={"length": 15, "base": 16},
    in_papercode={"length": 8, "base": 10},
    output_vocab=set(range(16)),  # digits D0..D15 for base up to 16
)
def generate_square_root(rng, batch_size, length, *, base:int = 10):
    """Integer square root. Input: digits of a number (LSD first) as D1..D(base-1).
    Digit encoding: D(digit+1) so that D0 is avoided for non-zero leading digit
    requirement. Actually let's keep it consistent: digits 0..base-1 map to D0..D(base-1).
    But original used 1-indexed (digits+1). Let's use D(digit) directly since D0 exists now.
    Wait, original ensured leading digit nonzero: digits[-1] = rng.integers(1, base).
    With unified vocab D0=0 exists, so digits[i] maps to D(digits[i]).

    Output: floor(sqrt(n)) digits as D1..D(base-1).
    """
    rng = _default_rng(rng)
    max_len = _max_digits_64(base)
    if length > max_len:
        raise ValueError(
            f"length={length} exceeds max {max_len} digits for 64-bit "
            f"numbers in base {base}")
    out_len = (length + 1) // 2

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, out_len), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []
    for b in range(batch_size):
        digits = rng.integers(0, base, size=length)
        digits[-1] = rng.integers(1, base)  # nonzero leading digit
        inputs[b] = digits  # D(digit) = digit value

        n = sum(int(digits[i]) * (base ** i) for i in range(length))
        root = math.isqrt(n)

        in_fmt.append(f"sqrt({n})")
        out_fmt.append(str(root))

        r = root
        for i in range(out_len):
            outputs[b, i] = r % base  # D(digit)
            r //= base

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [_strip_pad(outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def _generate_binary_addition(rng, batch_size, length):
    """Add two binary numbers. LSB first.

    Input: (batch_size, 2*length) — interleaved [a0,b0,a1,b1,...] using D0/D1.
    Output: (batch_size, length+1) — sum bits LSB first using D0/D1.
    """
    if length > _MAX_BINARY_ADD_BITS:
        raise ValueError(
            f"length={length} exceeds max {_MAX_BINARY_ADD_BITS} bits per number")
    a_bits = rng.integers(0, 2, size=(batch_size, length))
    b_bits = rng.integers(0, 2, size=(batch_size, length))

    # Interleave: D0 or D1 (token ID = bit value)
    inputs = np.zeros((batch_size, 2 * length), dtype=np.int64)
    inputs[:, 0::2] = a_bits  # D0=0, D1=1
    inputs[:, 1::2] = b_bits

    outputs = np.zeros((batch_size, length + 1), dtype=np.int64)
    carry = np.zeros(batch_size, dtype=np.int64)
    for i in range(length):
        s = a_bits[:, i] + b_bits[:, i] + carry
        outputs[:, i] = s % 2  # D0 or D1
        carry = s // 2
    outputs[:, length] = carry

    in_fmt = [f"{_bits_to_binstr(a_bits[b])} + {_bits_to_binstr(b_bits[b])}"
              for b in range(batch_size)]
    out_fmt = [f"{_bits_to_binstr(outputs[b])} ({_bits_to_int(outputs[b])})"
               for b in range(batch_size)]

    return {"input": [list(int(x) for x in inputs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# NOTE: Binary arithmetic tasks commented out — not in NSL TASK_NAMES (vocab removed binary tasks)
# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},  # D0, D1 (binary bits)
# )
def generate_8_bit_addition(rng, batch_size):
    """Add two 8-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_addition(rng, batch_size, 8)


# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},
# )
def generate_16_bit_addition(rng, batch_size):
    """Add two 16-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_addition(rng, batch_size, 16)


# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},
# )
def generate_32_bit_addition(rng, batch_size):
    """Add two 32-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_addition(rng, batch_size, 32)


# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},
# )
def generate_64_bit_addition(rng, batch_size):
    """Add two 64-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_addition(rng, batch_size, 64)


def _generate_binary_multiplication(rng, batch_size, length):
    """Multiply two binary numbers. LSB first.

    Input: (batch_size, 2*length) — interleaved using D0/D1.
    Output: (batch_size, 2*length) — product bits using D0/D1, PAD-padded.
    """
    if length > _MAX_BINARY_MUL_BITS:
        raise ValueError(
            f"length={length} exceeds max {_MAX_BINARY_MUL_BITS} bits per number")
    a_bits = rng.integers(0, 2, size=(batch_size, length))
    b_bits = rng.integers(0, 2, size=(batch_size, length))

    inputs = np.zeros((batch_size, 2 * length), dtype=np.int64)
    inputs[:, 0::2] = a_bits
    inputs[:, 1::2] = b_bits

    out_len = 2 * length
    outputs = np.zeros((batch_size, out_len), dtype=np.int64)
    for b in range(batch_size):
        a_val = sum(int(a_bits[b, i]) << i for i in range(length))
        b_val = sum(int(b_bits[b, i]) << i for i in range(length))
        product = a_val * b_val
        for i in range(out_len):
            outputs[b, i] = (product >> i) & 1  # D0 or D1

    in_fmt = [f"{_bits_to_binstr(a_bits[b])} * {_bits_to_binstr(b_bits[b])}"
              for b in range(batch_size)]
    out_fmt = [f"{_bits_to_binstr(outputs[b])} ({_bits_to_int(outputs[b])})"
               for b in range(batch_size)]

    return {"input": [list(int(x) for x in inputs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},
# )
def generate_8_bit_multiplication(rng, batch_size):
    """Multiply two 8-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_multiplication(rng, batch_size, 8)


# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},
# )
def generate_16_bit_multiplication(rng, batch_size):
    """Multiply two 16-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_multiplication(rng, batch_size, 16)


# @task(
#     category="arithmetic",
#     baseline={"batch_size": 32},
#     max={"batch_size": 32},
#     in_papercode={"batch_size": 32},
#     output_vocab={0, 1},
# )
def generate_32_bit_multiplication(rng, batch_size):
    """Multiply two 32-bit binary numbers. LSB first."""
    rng = _default_rng(rng)
    return _generate_binary_multiplication(rng, batch_size, 32)


# ============================================================================
# Data Processing Tasks
# ============================================================================

_MAX_PYEXEC_SEQLEN = 1_000


@task(
    category="data_processing",
    baseline={"length": 10, "max_value": 31},
    max={"length": 1000, "max_value": 31},
    in_papercode={"length": 50, "max_value": 31},
    output_vocab=set(range(1, 32)),  # D1..D31
)
def generate_sort(rng, batch_size, length, *, max_value=99):
    """Sort a sequence of integers in ascending order.

    Input: D1..D(max_value). Output: sorted D values.
    """
    rng = _default_rng(rng)
    _check_seqlen(2 * length, length=length)
    rng = _default_rng(rng)
    seqs = rng.integers(1, max_value + 1, size=(batch_size, length))
    outputs = np.sort(seqs, axis=1)

    in_fmt = ["[" + ", ".join(str(int(v)) for v in seqs[b]) + "]" for b in range(batch_size)]
    out_fmt = ["[" + ", ".join(str(int(v)) for v in outputs[b]) + "]" for b in range(batch_size)]

    return {"input": [list(int(x) for x in seqs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# NOTE: python_execution commented out — NSL is our own language, Python execution is redundant
# @task(
#     category="data_processing",
#     baseline={"length": 7, "max_val": 9, "max_iters": 5},
#     max={"length": 20, "max_val": 31, "max_iters": 10},
#     in_papercode={"length": 7, "max_val": 9, "max_iters": 5},
#     output_vocab=set(range(32)),  # D0..D31 (clamped result)
# )
def generate_python_execution(rng, batch_size, length, *, max_val=9, max_iters=5):
    """Predict output of simple Python-like programs.

    Token encoding: D0..D(max_val) for literals,
    PROG_ASSIGN, PROG_LOOP_START, PROG_LOOP_END, OP_ADD, OP_SUB, OP_MUL.
    Output: D(result) clamped to [0, max_val].
    """
    rng = _default_rng(rng)
    _check_seqlen(length + 1, limit=_MAX_PYEXEC_SEQLEN, length=length)
    rng = _default_rng(rng)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.zeros(batch_size, dtype=np.int64)

    op_sym = {0: "+", 1: "-", 2: "*"}
    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        init_val = rng.integers(0, max_val + 1)
        op_choice = rng.integers(0, 3)
        op_val = rng.integers(1, max_val + 1)
        n_iters = rng.integers(1, max_iters + 1)

        op_token = [V.OP_ADD, V.OP_SUB, V.OP_MUL][op_choice]
        tokens = [
            V.PROG_ASSIGN, int(init_val),           # D(init_val)
            V.PROG_LOOP_START, int(n_iters),         # D(n_iters)
            op_token, int(op_val),                    # D(op_val)
            V.PROG_LOOP_END,
        ]
        for i, t in enumerate(tokens):
            if i < length:
                inputs[b, i] = t

        # Execute
        x = init_val
        for _ in range(int(n_iters)):
            if op_choice == 0:
                x = x + op_val
            elif op_choice == 1:
                x = x - op_val
            else:
                x = x * op_val
        outputs[b] = np.clip(x, 0, max_val)

        code = (f"x = {int(init_val)}; "
                f"for _ in range({int(n_iters)}): "
                f"x = x {op_sym[int(op_choice)]} {int(op_val)}; "
                f"print(x)")
        in_fmt.append(code)
        out_fmt.append(str(int(outputs[b])))

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [[int(outputs[b])] for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# Mini-SHRDLU helpers
def _shrdlu_random_board(rng, grid_size, num_blocks):
    board = np.zeros((grid_size, grid_size), dtype=np.int64)
    blocks = list(range(1, num_blocks + 1))
    rng.shuffle(blocks)
    col_h = [0] * grid_size
    for block in blocks:
        avail = [c for c in range(grid_size) if col_h[c] < grid_size]
        if not avail:
            break
        col = avail[int(rng.integers(0, len(avail)))]
        board[col_h[col], col] = block
        col_h[col] += 1
    return board


def _shrdlu_board_key(board):
    return tuple(board.flatten())


def _shrdlu_board_moves(board, grid_size):
    col_top = {}
    col_h = [0] * grid_size
    for c in range(grid_size):
        for r in range(grid_size):
            if board[r, c] > 0:
                col_h[c] = r + 1
                col_top[c] = r

    results = []
    for src_col, src_row in col_top.items():
        block = int(board[src_row, src_col])
        for dst_col in range(grid_size):
            if dst_col == src_col:
                continue
            if col_h[dst_col] >= grid_size:
                continue
            nb = board.copy()
            nb[src_row, src_col] = 0
            nb[col_h[dst_col], dst_col] = block
            results.append(nb)
    return results


def _shrdlu_find_target(rng, board, grid_size, min_moves):
    start = _shrdlu_board_key(board)
    visited = {start}
    frontier = [board]
    candidates = []

    for depth in range(1, min_moves + 4):
        next_frontier = []
        for state in frontier:
            for nb in _shrdlu_board_moves(state, grid_size):
                key = _shrdlu_board_key(nb)
                if key not in visited:
                    visited.add(key)
                    next_frontier.append(nb)
                    if depth >= min_moves:
                        candidates.append(nb)
        frontier = next_frontier
        if not frontier:
            break
        if len(candidates) > 200:
            break

    if candidates:
        return candidates[int(rng.integers(0, len(candidates)))]
    return board


def _shrdlu_block_positions(board, grid_size):
    pos = {}
    for r in range(grid_size):
        for c in range(grid_size):
            if board[r, c] > 0:
                pos[int(board[r, c])] = (r, c)
    return pos


def _shrdlu_all_relations(positions):
    rels = []
    blocks = sorted(positions)
    for a in blocks:
        for bk in blocks:
            if a == bk:
                continue
            ra, ca = positions[a]
            rb, cb = positions[bk]
            if ca == cb and ra > rb:
                rels.append((a, "above", bk))
            if ca == cb and ra < rb:
                rels.append((a, "below", bk))
            if ca < cb:
                rels.append((a, "left", bk))
            if ca > cb:
                rels.append((a, "right", bk))
    return rels


def _shrdlu_prune_inverses(relations):
    seen = set()
    result = []
    for a, rel, bk in relations:
        if rel in ("above", "below"):
            key = ("v", min(a, bk), max(a, bk))
        else:
            key = ("h", min(a, bk), max(a, bk))
        if key not in seen:
            seen.add(key)
            result.append((a, rel, bk))
    return result


def _shrdlu_fmt_board(board, grid_size):
    rows = []
    for r in range(grid_size - 1, -1, -1):
        cells = [str(int(board[r, c])) if board[r, c] > 0 else "_"
                 for c in range(grid_size)]
        rows.append("[" + " ".join(cells) + "]")
    return " / ".join(rows)


@task(
    category="data_processing",
    baseline={"length": 20, "grid_size": 3, "num_blocks": 6, "min_moves": 2, "num_constraints": None},
    max={"length": 100, "grid_size": 5, "num_blocks": 10, "min_moves": 3, "num_constraints": None},
    in_papercode={"length": 30, "grid_size": 3, "num_blocks": 6, "min_moves": 2, "num_constraints": None},
    min_length=13,  # grid_size=3 requires board_cells + SEP + 1 constraint triple minimum
    output_vocab=set(range(11)),  # D0=empty, D1..D10 = block IDs
)
def generate_mini_shrdlu(rng, batch_size, length, *, grid_size=3, num_blocks=6,
                         min_moves=2, num_constraints=None):
    """Mini-SHRDLU blocks-world planning task.

    Token encoding:
      D0 = empty cell, D1..D(num_blocks) = block IDs,
      REL_ABOVE, REL_BELOW, REL_LEFT, REL_RIGHT = spatial relations,
      SEP = separator between board and constraints.

    Input: flattened board + SEP + constraint triples, PAD-padded.
    Output: flattened target board.
    """
    rng = _default_rng(rng)
    # Enforce minimum length (grid_size=3 requires board_cells + SEP + 1 constraint triple minimum)
    if length < 13:
        length = 13
    board_cells = grid_size * grid_size
    _check_seqlen(length + board_cells, length=length, grid_size=grid_size)
    rng = _default_rng(rng)
    max_constraints = max(1, (length - board_cells - 1) // 3)
    if num_constraints is None:
        num_constraints = min(4, max_constraints)
    else:
        num_constraints = min(num_constraints, max_constraints)

    rel_token = {"above": V.NAV_UP, "below": V.NAV_DOWN,
                 "left": V.NAV_LEFT, "right": V.NAV_RIGHT}
    rel_name = {"above": "above", "below": "below", "left": "left-of", "right": "right-of"}

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.zeros((batch_size, board_cells), dtype=np.int64)
    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        init_board = _shrdlu_random_board(rng, grid_size, num_blocks)
        target_board = _shrdlu_find_target(rng, init_board, grid_size, min_moves)

        init_pos = _shrdlu_block_positions(init_board, grid_size)
        target_pos = _shrdlu_block_positions(target_board, grid_size)

        init_rels = set(_shrdlu_all_relations(init_pos))
        target_rels = _shrdlu_all_relations(target_pos)

        novel = [r for r in target_rels if r not in init_rels]
        novel = _shrdlu_prune_inverses(novel)

        if len(novel) > num_constraints:
            idxs = rng.choice(len(novel), size=num_constraints, replace=False)
            selected = [novel[int(i)] for i in sorted(idxs)]
        else:
            selected = novel

        # Board values are already D(block_id) since block IDs are 0..num_blocks
        inputs[b, :board_cells] = init_board.flatten()  # D0=empty, D1..D(num_blocks)
        inputs[b, board_cells] = V.SEP
        for ci, (a, rel, bk) in enumerate(selected):
            off = board_cells + 1 + ci * 3
            if off + 2 < length:
                inputs[b, off] = a      # D(block_id)
                inputs[b, off + 1] = rel_token[rel]
                inputs[b, off + 2] = bk  # D(block_id)

        outputs[b] = target_board.flatten()

        in_fmt.append(
            _shrdlu_fmt_board(init_board, grid_size)
            + " | "
            + "; ".join(f"blk {a} {rel_name[rel]} blk {bk}"
                        for a, rel, bk in selected)
        )
        out_fmt.append(_shrdlu_fmt_board(target_board, grid_size))

    return {"input": [_strip_pad(inputs[b]) for b in range(batch_size)],
            "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# ============================================================================
# Graphs & Geometry Tasks
# ============================================================================

from scipy.spatial import ConvexHull, Delaunay, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, breadth_first_order

_MAX_ITEMS = 5_000


def _check_items(n, limit=_MAX_ITEMS, **ctx):
    if n > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(f"num items {n} exceeds limit {limit} ({detail})")


# Graph helpers
def _generate_connected_graph(rng, num_nodes, max_weight=None):
    """Generate a random connected graph as an adjacency matrix."""
    perm = rng.permutation(num_nodes)
    u = perm[:-1]
    v = perm[1:]

    weighted = max_weight is not None
    if weighted:
        w = rng.integers(1, max_weight + 1, size=num_nodes - 1)
    else:
        w = np.ones(num_nodes - 1, dtype=np.int64)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    adj[u, v] = w
    adj[v, u] = w

    num_extra = rng.integers(0, num_nodes)
    if num_extra > 0:
        extra_u = rng.integers(0, num_nodes, size=num_extra)
        extra_v = rng.integers(0, num_nodes, size=num_extra)
        mask = extra_u != extra_v
        if weighted:
            extra_w = rng.integers(1, max_weight + 1, size=num_extra)
        else:
            extra_w = np.ones(num_extra, dtype=np.int64)
        adj[extra_u[mask], extra_v[mask]] = extra_w[mask]
        adj[extra_v[mask], extra_u[mask]] = extra_w[mask]

    return adj


def _adj_to_edge_triples(adj):
    """Extract undirected edges as (u, v, w) triples. 1-indexed nodes."""
    rows, cols = np.nonzero(adj)
    mask = rows < cols
    rows = rows[mask] + 1
    cols = cols[mask] + 1
    weights = adj[rows - 1, cols - 1]
    return np.column_stack([rows, cols, weights])


def _edge_triples_to_flat(triples, length):
    """Flatten edge triples into a PAD-padded 1D array."""
    flat = np.full(length, V.PAD, dtype=np.int64)
    n = min(len(triples) * 3, length)
    if n > 0:
        flat[:n] = triples.flatten()[:n]
    return flat


def _fmt_edges(triples, weighted=True):
    if len(triples) == 0:
        return "{}"
    if weighted:
        edges = [f"{t[0]}-{t[1]}:{t[2]}" for t in triples]
    else:
        edges = [f"{t[0]}-{t[1]}" for t in triples]
    return "{" + ", ".join(edges) + "}"


# Geometry helpers
def _points_to_triples(coords):
    """Encode 2D points as (id, x, y) triples. 1-indexed IDs."""
    n = len(coords)
    ids = np.arange(1, n + 1).reshape(-1, 1)
    return np.hstack([ids, coords])


def _triples_to_flat(triples, length):
    """Flatten point triples into a PAD-padded 1D array."""
    flat = np.full(length, V.PAD, dtype=np.int64)
    n = min(len(triples) * 3, length)
    if n > 0:
        flat[:n] = triples.flatten()[:n]
    return flat


def _fmt_points(coords):
    return (
        "["
        + ", ".join(
            f"({int(coords[i, 0])},{int(coords[i, 1])})" for i in range(len(coords))
        )
        + "]"
    )


@task(
    category="graphs_geometry",
    baseline={"length": 30, "num_nodes": 10, "max_weight": 9},
    max={"length": 300, "num_nodes": 31, "max_weight": 31},
    in_papercode={"length": 60, "num_nodes": 10, "max_weight": 9},
    output_vocab=set(range(1, 32)),  # node IDs D1..D31
)
def generate_shortest_path(
    rng, batch_size: int, length: int, *, num_nodes: int = 10, max_weight: int = 9
):
    """Shortest path between random src and tgt nodes."""
    rng = _default_rng(rng)
    # Adjust to nearest multiple of 3 (required for edge triples)
    length = (length // 3) * 3
    if length < 3:
        length = 3
    _check_items(num_nodes, num_nodes=num_nodes)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, num_nodes), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        adj = _generate_connected_graph(rng, num_nodes, max_weight=max_weight)

        num_extra = num_nodes
        u_extra = rng.integers(0, num_nodes, size=num_extra)
        v_extra = rng.integers(0, num_nodes, size=num_extra)
        mask = u_extra != v_extra
        w_extra = rng.integers(1, max_weight + 1, size=num_extra)
        adj[u_extra[mask], v_extra[mask]] = w_extra[mask]
        adj[v_extra[mask], u_extra[mask]] = w_extra[mask]

        triples = _adj_to_edge_triples(adj)

        src = rng.integers(0, num_nodes)
        tgt = rng.integers(0, num_nodes)
        while tgt == src:
            tgt = rng.integers(0, num_nodes)

        flat_edges = _edge_triples_to_flat(triples, length - 3)
        inputs[b, : length - 3] = flat_edges
        inputs[b, length - 3] = src + 1
        inputs[b, length - 2] = tgt + 1
        inputs[b, length - 1] = V.PAD

        dist_matrix_result, predecessors = dijkstra(
            adj, directed=False, indices=[src], return_predecessors=True
        )

        path = []
        curr = tgt
        if dist_matrix_result[0, tgt] != np.inf:
            path = [tgt]
            while curr != src:
                curr = predecessors[0, curr]
                path.append(curr)
            path.reverse()

        if len(path) > 0:
            out_path = np.array(path, dtype=np.int64) + 1
            n_out = min(len(out_path), num_nodes)
            outputs[b, :n_out] = out_path[:n_out]

        if num_nodes <= 20:
            in_fmt.append(f"graph {_fmt_edges(triples)}, find path {src}->{tgt}")
            path_str = " -> ".join(str(n) for n in path)
            out_fmt.append(f"path: {path_str}")
        else:
            in_fmt.append("graph (large)")
            out_fmt.append("path (large)")

    return {
        "input": [_strip_pad(inputs[b]) for b in range(batch_size)],
        "output": [_strip_pad(outputs[b]) for b in range(batch_size)],
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


@task(
    category="graphs_geometry",
    baseline={"length": 30, "num_nodes": 10, "max_weight": 9},
    max={"length": 300, "num_nodes": 31, "max_weight": 31},
    in_papercode={"length": 60, "num_nodes": 10, "max_weight": 9},
    output_vocab=set(range(1, 32)),  # node IDs D1..D31 + weights D1..D31
)
def generate_mst_prim(rng, batch_size: int, length: int, *, num_nodes: int = 10, max_weight: int = 9):
    """Minimum spanning tree of a weighted graph."""
    rng = _default_rng(rng)
    # Adjust to nearest multiple of 3 (required for edge triples)
    length = (length // 3) * 3
    if length < 3:
        length = 3
    _check_items(num_nodes, num_nodes=num_nodes)

    mst_len = 3 * (num_nodes - 1)
    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, mst_len), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        adj = _generate_connected_graph(rng, num_nodes, max_weight=max_weight)
        triples = _adj_to_edge_triples(adj)
        inputs[b] = _edge_triples_to_flat(triples, length)

        mst_sparse = minimum_spanning_tree(adj)
        mst = mst_sparse.toarray().astype(np.int64)
        mst = np.maximum(mst, mst.T)
        mst_triples = _adj_to_edge_triples(mst)
        outputs[b] = _edge_triples_to_flat(mst_triples, mst_len)

        if num_nodes <= 20:
            in_fmt.append(f"graph {_fmt_edges(triples)}")
            total_w = sum(int(t[2]) for t in mst_triples)
            out_fmt.append(f"MST {_fmt_edges(mst_triples)} (weight={total_w})")
        else:
            in_fmt.append("graph (large)")
            out_fmt.append("MST (large)")

    return {
        "input": [_strip_pad(inputs[b]) for b in range(batch_size)],
        "output": [_strip_pad(outputs[b]) for b in range(batch_size)],
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


@task(
    category="graphs_geometry",
    baseline={"length": 30, "num_nodes": 10},
    max={"length": 300, "num_nodes": 31},
    in_papercode={"length": 60, "num_nodes": 10},
    output_vocab=set(range(1, 32)),  # BFS ranks 1..31
)
def generate_graph_traversal(rng, batch_size: int, length: int, *, num_nodes: int = 10):
    """BFS traversal order on an unweighted graph."""
    rng = _default_rng(rng)
    # Adjust to nearest multiple of 3 (required for edge triples)
    length = (length // 3) * 3
    if length < 3:
        length = 3
    _check_items(num_nodes, num_nodes=num_nodes)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.zeros((batch_size, num_nodes), dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        adj = _generate_connected_graph(rng, num_nodes, max_weight=None)
        triples = _adj_to_edge_triples(adj)
        inputs[b] = _edge_triples_to_flat(triples, length)

        node_order, predecessors = breadth_first_order(adj, i_start=0, directed=False)

        rank = np.zeros(num_nodes, dtype=np.int64)
        rank[node_order] = np.arange(1, len(node_order) + 1)
        outputs[b] = rank

        if num_nodes <= 20:
            in_fmt.append(f"graph {_fmt_edges(triples, weighted=False)}, src=0")
            out_fmt.append(f"BFS order: {' -> '.join(str(n) for n in node_order)}")
        else:
            in_fmt.append("graph (large), src=0")
            out_fmt.append("BFS order (large)")

    return {
        "input": [_strip_pad(inputs[b]) for b in range(batch_size)],
        "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


@task(
    category="graphs_geometry",
    baseline={"length": 30, "num_cities": 10, "coord_scale": 31},
    max={"length": 300, "num_cities": 31, "coord_scale": 31},
    in_papercode={"length": 60, "num_cities": 10, "coord_scale": 31},
    output_vocab=set(range(1, 32)),  # city IDs D1..D31
)
def generate_tsp(rng, batch_size, length, *, num_cities=None, coord_scale=100):
    """TSP nearest neighbor heuristic.

    Input: point triples [id1,x1,y1, ...], 1-indexed IDs.
    Output: tour as 1-indexed city IDs, PAD-padded.
    """
    rng = _default_rng(rng)
    # Adjust to nearest multiple of 3 (required for point triples)
    length = (length // 3) * 3
    if length < 3:
        length = 3
    if num_cities is None:
        num_cities = length // 3
    _check_items(num_cities, num_cities=num_cities)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, num_cities), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        coords = rng.integers(0, coord_scale, size=(num_cities, 2))
        triples = _points_to_triples(coords)
        inputs[b] = _triples_to_flat(triples, length)

        dists = distance_matrix(coords, coords)

        current = 0
        tour = [0]
        dists[:, 0] = np.inf

        for _ in range(num_cities - 1):
            next_city = np.argmin(dists[current])
            tour.append(int(next_city))
            dists[:, next_city] = np.inf
            current = next_city

        outputs[b] = np.array(tour, dtype=np.int64) + 1

        if num_cities <= 20:
            in_fmt.append(f"cities {_fmt_points(coords)}")
            out_fmt.append(f"tour: {' -> '.join(str(c + 1) for c in tour)}")
        else:
            in_fmt.append("cities (large)")
            out_fmt.append("tour (large)")

    return {
        "input": [_strip_pad(inputs[b]) for b in range(batch_size)],
        "output": [_strip_pad(outputs[b]) for b in range(batch_size)],
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


@task(
    category="graphs_geometry",
    baseline={"length": 30, "num_points": 10, "coord_scale": 31},
    max={"length": 300, "num_points": 50, "coord_scale": 31},
    in_papercode={"length": 60, "num_points": 10, "coord_scale": 31},
    output_vocab={V.TRUE, V.FALSE},  # per-point hull membership
)
def generate_convex_hull(
    rng, batch_size, length, *, num_points=None, coord_scale=100
):
    """Convex Hull.

    Input: point triples [id1,x1,y1, ...], 1-indexed IDs.
    Output: binary mask — TRUE if on hull, FALSE otherwise.
    """
    rng = _default_rng(rng)
    # Adjust to nearest multiple of 3 (required for point triples)
    length = (length // 3) * 3
    if length < 3:
        length = 3
    if num_points is None:
        num_points = length // 3
    _check_items(num_points, num_points=num_points)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, num_points), V.FALSE, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        points = rng.integers(0, coord_scale, size=(num_points, 2))
        triples = _points_to_triples(points)
        inputs[b] = _triples_to_flat(triples, length)

        if num_points >= 3:
            try:
                hull = ConvexHull(points)
                outputs[b, hull.vertices] = V.TRUE
                hull_indices = sorted(hull.vertices)
            except Exception:
                hull_indices = []
        else:
            outputs[b, :] = V.TRUE
            hull_indices = list(range(num_points))

        if num_points <= 20:
            in_fmt.append(f"points {_fmt_points(points)}")
            hull_pts = [
                f"p{i + 1}({int(points[i, 0])},{int(points[i, 1])})"
                for i in hull_indices
            ]
            out_fmt.append(f"hull: {{{', '.join(hull_pts)}}}")
        else:
            in_fmt.append("points (large)")
            out_fmt.append("hull (large)")

    return {
        "input": [_strip_pad(inputs[b]) for b in range(batch_size)],
        "output": [list(int(x) for x in outputs[b]) for b in range(batch_size)],
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


@task(
    category="graphs_geometry",
    baseline={"length": 30, "num_points": 10, "coord_scale": 31, "max_triangles": None},
    max={"length": 300, "num_points": 31, "coord_scale": 31, "max_triangles": None},
    in_papercode={"length": 60, "num_points": 10, "coord_scale": 31, "max_triangles": None},
    output_vocab=set(range(1, 32)),  # vertex IDs D1..D31
)
def generate_delaunay(
    rng,
    batch_size,
    length,
    *,
    num_points=None,
    coord_scale=100,
    max_triangles=None,
):
    """Delaunay triangulation.

    Input: point triples [id1,x1,y1, ...], 1-indexed IDs.
    Output: triangle vertex triples (1-indexed), PAD-padded.
    """
    rng = _default_rng(rng)
    # Adjust to nearest multiple of 3 (required for point triples)
    length = (length // 3) * 3
    if length < 3:
        length = 3
    if num_points is None:
        num_points = length // 3
    _check_items(num_points, num_points=num_points)
    if max_triangles is None:
        max_triangles = max(1, 2 * num_points - 5)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, max_triangles * 3), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        points = rng.integers(0, coord_scale, size=(num_points, 2))
        triples = _points_to_triples(points)
        inputs[b] = _triples_to_flat(triples, length)

        triangles = []
        if num_points >= 3:
            try:
                delaunay = Delaunay(points)
                triangles = delaunay.simplices
            except Exception:
                pass

        n_tris = min(len(triangles), max_triangles)
        if n_tris > 0:
            outputs[b, : n_tris * 3] = np.array(triangles[:n_tris]).flatten() + 1

        if num_points <= 20:
            in_fmt.append(f"points {_fmt_points(points)}")
            tri_strs = [
                f"({t[0] + 1},{t[1] + 1},{t[2] + 1})" for t in triangles[:n_tris]
            ]
            out_fmt.append(f"triangles: [{', '.join(tri_strs)}]")
        else:
            in_fmt.append("points (large)")
            out_fmt.append("triangles (large)")

    return {
        "input": [_strip_pad(inputs[b]) for b in range(batch_size)],
        "output": [_strip_pad(outputs[b]) for b in range(batch_size)],
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }
