"""Regular tasks — solvable by finite automata.

Each generator returns {"input": np.ndarray, "output": np.ndarray,
"input_formatted": list[str], "output_formatted": list[str]}.
Input shape: (batch_size, length) — integer-encoded sequences.
Output shape: (batch_size,) — classification labels.
"""

import numpy as np

from .. import vocab as V

_MAX_SEQLEN = 10_000


def _default_rng(rng):
    if rng is None:
        return np.random.default_rng()
    return rng


def _check_seqlen(total, limit=_MAX_SEQLEN, **ctx):
    if total > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(
            f"total seqlen {total} exceeds limit {limit} ({detail})")


_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def generate_even_pairs(rng, batch_size, length, *, vocab_size=2):
    """Check if symbols appear in identical adjacent pairs.

    Vocab: digits D1..D(vocab_size). Label TRUE/FALSE.
    """
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

    return {"input": seqs, "output": labels,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_parity_check(rng, batch_size, length, *, symbol=1, vocab_size=2):
    """Determine if the count of `symbol` in the sequence is even or odd.

    Output: TRUE (even) or FALSE (odd). Wait — original was 0=even, 1=odd.
    Let's keep: TRUE = odd count, FALSE = even count to match original label=1 means odd.
    Actually original: labels = (counts % 2) where 1=odd, 0=even.
    Map: 1 -> TRUE (odd), 0 -> FALSE (even).
    """
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

    return {"input": seqs, "output": labels,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_cycle_navigation(rng, batch_size, length, *, num_states=5):
    """Navigate a cycle of `num_states` states. NAV_FWD=forward, NAV_BWD=backward.

    Output is the final state as D(state).
    """
    _check_seqlen(length + 1, length=length)
    rng = _default_rng(rng)
    # Choose FWD or BWD
    choices = rng.integers(0, 2, size=(batch_size, length))
    seqs = np.where(choices == 0, V.NAV_FWD, V.NAV_BWD).astype(np.int64)

    pos = np.zeros(batch_size, dtype=np.int64)
    for t in range(length):
        step = np.where(seqs[:, t] == V.NAV_FWD, 1, -1)
        pos = (pos + step) % num_states

    arrow = {V.NAV_FWD: "->", V.NAV_BWD: "<-"}
    in_fmt = [" ".join(arrow[int(seqs[b, t])] for t in range(length)) for b in range(batch_size)]
    out_fmt = [f"state {pos[b]}" for b in range(batch_size)]

    return {"input": seqs, "output": pos,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_modular_arithmetic(rng, batch_size, length, *, modulus=5):
    """Evaluate simple arithmetic expressions (no brackets) under modulus.

    Encoding: operands D0..D(modulus-1), operators OP_ADD/OP_SUB/OP_MUL.
    Output: D(result).
    """
    _check_seqlen(length + 1, length=length)
    rng = _default_rng(rng)
    if length % 2 == 0:
        raise ValueError("length must be odd for alternating operand/operator format")

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

    return {"input": seqs, "output": result.astype(np.int64),
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# ---- Vocab info functions ----

def _even_pairs_vocab_info(vocab_size=2, **_kw):
    inp = set(range(1, vocab_size + 1))
    out = {V.TRUE, V.FALSE}
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _parity_check_vocab_info(vocab_size=2, **_kw):
    inp = set(range(1, vocab_size + 1))
    out = {V.TRUE, V.FALSE}
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _cycle_navigation_vocab_info(num_states=5, **_kw):
    inp = {V.NAV_FWD, V.NAV_BWD}
    out = set(range(0, num_states))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _modular_arithmetic_vocab_info(modulus=5, **_kw):
    inp = set(range(0, modulus)) | {V.OP_ADD, V.OP_SUB, V.OP_MUL}
    out = set(range(0, modulus))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


VOCAB_INFO = {
    "even_pairs": _even_pairs_vocab_info,
    "parity_check": _parity_check_vocab_info,
    "cycle_navigation": _cycle_navigation_vocab_info,
    "modular_arithmetic": _modular_arithmetic_vocab_info,
}
