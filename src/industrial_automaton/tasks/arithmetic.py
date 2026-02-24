import math
import numpy as np

from industrial_automaton import vocab as V

_MAX_BINARY_ADD_BITS = 64
_MAX_BINARY_MUL_BITS = 32

def _max_digits_64(base):
    """Max number of digits so that a base-`base` number fits in 64 bits."""
    return int(64 * math.log(2) / math.log(base))

def generate_square_root(rng, batch_size, length, *, base:int = 10):
    """Integer square root. Input: digits of a number (LSD first) as D1..D(base-1).
    Digit encoding: D(digit+1) so that D0 is avoided for non-zero leading digit
    requirement. Actually let's keep it consistent: digits 0..base-1 map to D0..D(base-1).
    But original used 1-indexed (digits+1). Let's use D(digit) directly since D0 exists now.
    Wait, original ensured leading digit nonzero: digits[-1] = rng.integers(1, base).
    With unified vocab D0=0 exists, so digits[i] maps to D(digits[i]).
    
    Output: floor(sqrt(n)) digits as D1..D(base-1).
    """
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}



def _bits_to_binstr(bits):
    """Convert LSB-first bit array to human-readable MSB-first binary string."""
    s = "".join(str(int(b)) for b in reversed(bits))
    return s.lstrip("0") or "0"


def _bits_to_int(bits):
    return sum(int(bits[i]) << i for i in range(len(bits)))


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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}

def generate_8_bit_addition(rng, batch_size):
    """Add two 8-bit binary numbers. LSB first."""
    return _generate_binary_addition(rng, batch_size, 8)

def generate_16_bit_addition(rng, batch_size):
    """Add two 16-bit binary numbers. LSB first."""
    return _generate_binary_addition(rng, batch_size, 16)

def generate_32_bit_addition(rng, batch_size):
    """Add two 32-bit binary numbers. LSB first."""
    return _generate_binary_addition(rng, batch_size, 32)

def generate_64_bit_addition(rng, batch_size):
    """Add two 64-bit binary numbers. LSB first."""
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}

def generate_8_bit_multiplication(rng, batch_size):
    """Multiply two 8-bit binary numbers. LSB first."""
    return _generate_binary_multiplication(rng, batch_size, 8)

def generate_16_bit_multiplication(rng, batch_size):
    """Multiply two 16-bit binary numbers. LSB first."""
    return _generate_binary_multiplication(rng, batch_size, 16)

def generate_32_bit_multiplication(rng, batch_size):
    """Multiply two 32-bit binary numbers. LSB first."""
    return _generate_binary_multiplication(rng, batch_size, 32)


# ---- Vocab info functions ----

def _binary_addition_vocab_info(**_kw):
    toks = {0, 1}  # D0, D1
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _binary_multiplication_vocab_info(**_kw):
    toks = {0, 1}
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _square_root_vocab_info(base=10, **_kw):
    inp = set(range(0, base)) | {V.PAD}
    out = set(range(0, base)) | {V.PAD}
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


VOCAB_INFO = {
    "binary_addition": _binary_addition_vocab_info,
    "binary_multiplication": _binary_multiplication_vocab_info,
    "square_root": _square_root_vocab_info,
}