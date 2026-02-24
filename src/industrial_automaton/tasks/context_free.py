"""Context-free / DCF tasks — require stack memory.

Each generator returns {"input": np.ndarray, "output": np.ndarray,
"input_formatted": list[str], "output_formatted": list[str]}.
"""

import numpy as np

from .. import vocab as V

_MAX_SEQLEN = 10_000
_MAX_DYCK_N = 20  # capped by vocab layout (20 open + 20 close tokens)


def _default_rng(rng):
    if rng is None:
        return np.random.default_rng()
    return rng


def _check_seqlen(total, limit=_MAX_SEQLEN, **ctx):
    if total > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(
            f"total seqlen {total} exceeds limit {limit} ({detail})")


def generate_stack_manipulation(rng, batch_size: int, length: int, *, vocab_size: int = 4):
    """Execute push/pop instructions, output top-of-stack after each step.

    Encoding: D1..D(vocab_size) = push that value, STACK_POP = pop.
    Output: top-of-stack value as D(val) after each instruction (V.PAD if stack empty).
    Shapes: input (batch_size, length), output (batch_size, length).
    """
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

    return {"input": seqs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_reverse_string(rng, batch_size, length, *, vocab_size: int = 8):
    """Recall the input sequence in reverse order.

    Input: D1..D(vocab_size). Output: same tokens reversed.
    """
    _check_seqlen(2 * length, length=length)
    rng = _default_rng(rng)
    seqs = rng.integers(1, vocab_size + 1, size=(batch_size, length))
    outputs = seqs[:, ::-1].copy()

    alpha = "abcdefghijklmnopqrstuvwxyz"
    in_fmt = [" ".join(alpha[t - 1] for t in seqs[b]) for b in range(batch_size)]
    out_fmt = [" ".join(alpha[t - 1] for t in outputs[b]) for b in range(batch_size)]

    return {"input": seqs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_nested_modular_arithmetic(rng, batch_size: int, length: int, *, modulus: int=5, max_depth: int=3):
    """Modular arithmetic with nested parentheses.

    Encoding: operands D0..D(modulus-1), operators OP_ADD/OP_SUB/OP_MUL,
    OPEN_PAREN, CLOSE_PAREN. Output: D(result).
    Shapes: input (batch_size, length) padded with PAD, output (batch_size,).
    """
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
               V.OPEN_PAREN: "(", V.CLOSE_PAREN: ")"}
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

    return {"input": inputs, "output": np.array(all_results, dtype=np.int64),
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
        tokens = [V.OPEN_PAREN] + left_tokens + [op_token] + right_tokens + [V.CLOSE_PAREN]
    else:
        tokens = left_tokens + [op_token] + right_tokens

    return tokens, result


def generate_solve_equation(rng, batch_size, length, *, max_val=9):
    """Find the value of x in a simple linear equation: a * x + b = c.

    Encoding: D(a), OP_MUL, VAR_X, OP_ADD, D(b), OP_EQ, D(c).
    Output: D(x).
    """
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


_BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


def generate_dyck_n(rng, batch_size, length, *, n: int = 2):
    """Validate balanced parentheses with n bracket types.

    Bracket types: OPEN(i), CLOSE(i) for i in [0, n-1].
    Output: TRUE if balanced, FALSE otherwise.
    """
    if n > _MAX_DYCK_N:
        raise ValueError(f"n={n} exceeds max {_MAX_DYCK_N}")
    _check_seqlen(length + 1, length=length, n=n)
    rng = _default_rng(rng)
    seqs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    labels = np.full(batch_size, V.FALSE, dtype=np.int64)

    for b in range(batch_size):
        if rng.random() < 0.5:
            tokens = _gen_dyck_balanced(rng, n, length)
            for t, tok in enumerate(tokens):
                if t < length:
                    seqs[b, t] = tok
            if len(tokens) == length:
                labels[b] = V.TRUE
            # else stays FALSE (padded / doesn't fill)
        else:
            for t in range(length):
                # Random bracket token
                bracket_idx = rng.integers(0, 2 * n)
                if bracket_idx < n:
                    seqs[b, t] = V.OPEN(int(bracket_idx))
                else:
                    seqs[b, t] = V.CLOSE(int(bracket_idx - n))
            labels[b] = V.TRUE if _check_dyck_balanced(seqs[b], n) else V.FALSE

    # Formatting
    if n <= len(_BRACKET_PAIRS):
        brackets = _BRACKET_PAIRS[:n]
        in_fmt = []
        for b in range(batch_size):
            chars = []
            for tok in seqs[b]:
                if tok == V.PAD:
                    break
                tok = int(tok)
                for i in range(n):
                    if tok == V.OPEN(i):
                        chars.append(brackets[i][0])
                        break
                    if tok == V.CLOSE(i):
                        chars.append(brackets[i][1])
                        break
            in_fmt.append("".join(chars))
    else:
        in_fmt = []
        for b in range(batch_size):
            parts = []
            for tok in seqs[b]:
                if tok == V.PAD:
                    break
                tok = int(tok)
                for i in range(n):
                    if tok == V.OPEN(i):
                        parts.append(f"o{i}")
                        break
                    if tok == V.CLOSE(i):
                        parts.append(f"c{i}")
                        break
            in_fmt.append(" ".join(parts))

    out_fmt = ["balanced" if labels[b] == V.TRUE else "unbalanced" for b in range(batch_size)]

    return {"input": seqs, "output": labels,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def _gen_dyck_balanced(rng, n, target_length):
    """Generate a balanced Dyck-n sequence up to target_length."""
    tokens = []
    stack = []
    while len(tokens) < target_length:
        remaining = target_length - len(tokens)
        if remaining == len(stack):
            if stack:
                bracket_type = stack.pop()
                tokens.append(V.CLOSE(bracket_type))
            else:
                break
        elif stack and rng.random() < 0.4:
            bracket_type = stack.pop()
            tokens.append(V.CLOSE(bracket_type))
        else:
            bracket_type = int(rng.integers(0, n))
            stack.append(bracket_type)
            tokens.append(V.OPEN(bracket_type))
    return tokens


def _check_dyck_balanced(seq, n):
    """Check if a sequence is a valid Dyck-n word."""
    stack = []
    for token in seq:
        token = int(token)
        if token == V.PAD:
            continue
        # Check if it's an open bracket
        is_open = False
        for i in range(n):
            if token == V.OPEN(i):
                stack.append(i)
                is_open = True
                break
        if is_open:
            continue
        # Check if it's a close bracket
        for i in range(n):
            if token == V.CLOSE(i):
                if not stack or stack[-1] != i:
                    return False
                stack.pop()
                break
    return len(stack) == 0


# ---- Vocab info functions ----

def _stack_manipulation_vocab_info(vocab_size=4, **_kw):
    inp = set(range(1, vocab_size + 1)) | {V.STACK_POP}
    out = set(range(1, vocab_size + 1)) | {V.PAD}  # PAD for empty stack
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _reverse_string_vocab_info(vocab_size=8, **_kw):
    toks = set(range(1, vocab_size + 1))
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _nested_modular_arithmetic_vocab_info(modulus=5, **_kw):
    inp = set(range(0, modulus)) | {V.OP_ADD, V.OP_SUB, V.OP_MUL, V.OPEN_PAREN, V.CLOSE_PAREN, V.PAD}
    out = set(range(0, modulus))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _solve_equation_vocab_info(max_val=9, **_kw):
    inp = set(range(0, max_val + 1)) | {V.VAR_X, V.OP_MUL, V.OP_ADD, V.OP_EQ, V.PAD}
    out = set(range(0, max_val + 1))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _dyck_n_vocab_info(n=2, **_kw):
    inp = {V.OPEN(i) for i in range(n)} | {V.CLOSE(i) for i in range(n)} | {V.PAD}
    out = {V.TRUE, V.FALSE}
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


VOCAB_INFO = {
    "stack_manipulation": _stack_manipulation_vocab_info,
    "reverse_string": _reverse_string_vocab_info,
    "nested_modular_arithmetic": _nested_modular_arithmetic_vocab_info,
    "solve_equation": _solve_equation_vocab_info,
    "dyck_n": _dyck_n_vocab_info,
}
