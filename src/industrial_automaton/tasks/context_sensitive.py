"""Context-sensitive & algorithmic tasks — require tape/matrix memory.

Each generator returns {"input": np.ndarray, "output": np.ndarray,
"input_formatted": list[str], "output_formatted": list[str]}.
Variable output lengths use V.PAD for padding.
"""

import math
import numpy as np

from .. import vocab as V

_MAX_SEQLEN = 10_000
_MAX_COUNT_N = 500
_MAX_BINARY_ADD_BITS = 64
_MAX_BINARY_MUL_BITS = 32
_MAX_REPEAT_COPY_LEN = 2_000
_MAX_REPEAT_COPY_N = 50

def _check_seqlen(total, limit=_MAX_SEQLEN, **ctx):
    if total > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(
            f"total seqlen {total} exceeds limit {limit} ({detail})")


def _max_digits_64(base):
    """Max number of digits so that a base-`base` number fits in 64 bits."""
    return int(64 * math.log(2) / math.log(base))


_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _seq_to_letters(seq):
    return " ".join(_ALPHA[t - 1] for t in seq if 1 <= t <= 26)


def generate_duplicate_string(rng, batch_size, length, *, vocab_size=8):
    """Duplicate input: w -> ww.

    Input: (batch_size, length) with D1..D(vocab_size).
    Output: (batch_size, 2*length).
    """
    _check_seqlen(3 * length, length=length)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    outputs = np.concatenate([seqs, seqs], axis=1)

    in_fmt = [_seq_to_letters(seqs[b]) for b in range(batch_size)]
    out_fmt = [_seq_to_letters(outputs[b]) for b in range(batch_size)]

    return {"input": seqs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_repeat_copy_n(rng, batch_size, length, *, max_n=5, vocab_size=8):
    """Reproduce an input string N times.

    Input: [D(n), tok1, tok2, ...] where n is repeat count.
    Output: pattern repeated n times, PAD-padded.
    """
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_deduplicate_inputs(rng, batch_size, length, *, repeat: int = 3, vocab_size: int = 8):
    """Filter redundant stream: each symbol repeated `repeat` times -> unique symbols.

    Input: (batch_size, length * repeat) with D1..D(vocab_size).
    Output: (batch_size, length).
    """
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

    return {"input": expanded, "output": unique,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_associative_recall(rng, batch_size, length, *, num_pairs=None, vocab_size=8):
    """Retrieve value for a queried key from key-value pairs.

    Input layout: [k1, v1, k2, v2, ..., kN, vN, query_key, PAD].
    Output: D(value).
    """
    _check_seqlen(length + 1, length=length)
    if num_pairs is None:
        num_pairs = max(1, (length - 2) // 2)

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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_missing_duplicate(rng, batch_size, length, *, vocab_size=None):
    """Identify the missing element from a duplicated sequence.

    Input: (batch_size, 2*length - 1) with D1..D(vocab_size).
    Output: D(missing_element).
    """
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_odds_first(rng, batch_size, length, *, vocab_size=10):
    """Reorder so odd-valued elements come first, then even-valued.

    Input/Output: D1..D(vocab_size).
    """
    _check_seqlen(2 * length, length=length)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    outputs = np.zeros_like(seqs)

    for b in range(batch_size):
        odd_vals = seqs[b, seqs[b] % 2 == 1]
        even_vals = seqs[b, seqs[b] % 2 == 0]
        outputs[b] = np.concatenate([odd_vals, even_vals])

    in_fmt = [" ".join(str(int(t)) for t in seqs[b]) for b in range(batch_size)]
    out_fmt = [" ".join(str(int(t)) for t in outputs[b]) for b in range(batch_size)]

    return {"input": seqs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_count_n(rng, batch_size, length, *, n: int = 3):
    """Validate s1^k s2^k ... sn^k: n symbol types each appearing k times.

    Encoding: D1..D(n). Output: TRUE/FALSE.
    """
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

    return {"input": seqs, "output": labels,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_n_back(rng, batch_size, length, *, n=None, vocab_size=8):
    """N-back temporal recall task.

    Input layout: [s1, s2, ..., sL, HASH_SEP, D(n)].
    Output: D(symbol) — the symbol n positions before the last.
    """
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
    inputs[:, length] = V.HASH_SEP
    inputs[:, length + 1] = ns  # D(n)

    outputs = np.array([seqs[b, length - 1 - ns[b]] for b in range(batch_size)],
                       dtype=np.int64)

    in_fmt = [_seq_to_letters(seqs[b]) + f" # {ns[b]}"
              for b in range(batch_size)]
    out_fmt = [_ALPHA[outputs[b] - 1] for b in range(batch_size)]

    return {"input": inputs, "output": outputs,
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


# ---- Vocab info functions ----

def _duplicate_string_vocab_info(vocab_size=8, **_kw):
    toks = set(range(1, vocab_size + 1))
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _repeat_copy_n_vocab_info(max_n=5, vocab_size=8, **_kw):
    inp = set(range(0, max_n + 1)) | set(range(1, vocab_size + 1)) | {V.PAD}
    out = set(range(1, vocab_size + 1)) | {V.PAD}
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _deduplicate_inputs_vocab_info(vocab_size=8, **_kw):
    toks = set(range(1, vocab_size + 1))
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _associative_recall_vocab_info(vocab_size=8, **_kw):
    toks = set(range(1, vocab_size + 1)) | {V.PAD}
    out = set(range(1, vocab_size + 1))
    return {"input_tokens": toks, "output_tokens": out, "output_mask": V._make_mask(out)}


def _missing_duplicate_vocab_info(vocab_size=None, length=10, **_kw):
    if vocab_size is None:
        vocab_size = length + 2
    toks = set(range(1, vocab_size + 1)) | {V.PAD}
    out = set(range(1, vocab_size + 1))
    return {"input_tokens": toks, "output_tokens": out, "output_mask": V._make_mask(out)}


def _odds_first_vocab_info(vocab_size=10, **_kw):
    toks = set(range(1, vocab_size + 1))
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _count_n_vocab_info(n=3, **_kw):
    inp = set(range(1, n + 1))
    out = {V.TRUE, V.FALSE}
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _n_back_vocab_info(vocab_size=8, **_kw):
    inp = set(range(1, vocab_size + 1)) | {V.HASH_SEP, V.PAD}
    out = set(range(1, vocab_size + 1))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


VOCAB_INFO = {
    "duplicate_string": _duplicate_string_vocab_info,
    "repeat_copy_n": _repeat_copy_n_vocab_info,
    "deduplicate_inputs": _deduplicate_inputs_vocab_info,
    "associative_recall": _associative_recall_vocab_info,
    "missing_duplicate": _missing_duplicate_vocab_info,
    "odds_first": _odds_first_vocab_info,
    "count_n": _count_n_vocab_info,
    "n_back": _n_back_vocab_info,
}
