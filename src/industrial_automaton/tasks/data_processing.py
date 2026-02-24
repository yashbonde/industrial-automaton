"""Data processing & code logic tasks.

Each generator returns {"input": np.ndarray, "output": np.ndarray,
"input_formatted": list[str], "output_formatted": list[str]}.
"""

import numpy as np

from .. import vocab as V

_MAX_SEQLEN = 10_000
_MAX_PYEXEC_SEQLEN = 1_000


def _default_rng(rng):
    if rng is None:
        return np.random.default_rng()
    return rng


def _check_seqlen(total, limit=_MAX_SEQLEN, **ctx):
    if total > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(
            f"total seqlen {total} exceeds limit {limit} ({detail})")


def generate_sort(rng, batch_size, length, *, max_value=99):
    """Sort a sequence of integers in ascending order.

    Input: D1..D(max_value). Output: sorted D values.
    """
    _check_seqlen(2 * length, length=length)
    rng = _default_rng(rng)
    seqs = rng.integers(1, max_value + 1, size=(batch_size, length))
    outputs = np.sort(seqs, axis=1)

    in_fmt = ["[" + ", ".join(str(int(v)) for v in seqs[b]) + "]" for b in range(batch_size)]
    out_fmt = ["[" + ", ".join(str(int(v)) for v in outputs[b]) + "]" for b in range(batch_size)]

    return {"input": seqs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


def generate_python_execution(rng, batch_size, length, *, max_val=9, max_iters=5):
    """Predict output of simple Python-like programs.

    Token encoding: D0..D(max_val) for literals,
    PROG_ASSIGN, PROG_LOOP_START, PROG_LOOP_END, OP_ADD, OP_SUB, OP_MUL.
    Output: D(result) clamped to [0, max_val].
    """
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


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
    board_cells = grid_size * grid_size
    _check_seqlen(length + board_cells, length=length, grid_size=grid_size)
    rng = _default_rng(rng)
    max_constraints = max(1, (length - board_cells - 1) // 3)
    if num_constraints is None:
        num_constraints = min(4, max_constraints)
    else:
        num_constraints = min(num_constraints, max_constraints)

    rel_token = {"above": V.REL_ABOVE, "below": V.REL_BELOW,
                 "left": V.REL_LEFT, "right": V.REL_RIGHT}
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

    return {"input": inputs, "output": outputs,
            "input_formatted": in_fmt, "output_formatted": out_fmt}


# --------------- Mini-SHRDLU helpers ---------------

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


# ---- Vocab info functions ----

def _sort_vocab_info(max_value=99, **_kw):
    toks = set(range(1, max_value + 1))
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _python_execution_vocab_info(max_val=9, **_kw):
    inp = set(range(0, max_val + 1)) | {V.PROG_ASSIGN, V.PROG_LOOP_START, V.PROG_LOOP_END,
                                          V.OP_ADD, V.OP_SUB, V.OP_MUL, V.PAD}
    out = set(range(0, max_val + 1))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


def _mini_shrdlu_vocab_info(num_blocks=6, **_kw):
    inp = set(range(0, num_blocks + 1)) | {V.REL_ABOVE, V.REL_BELOW, V.REL_LEFT, V.REL_RIGHT,
                                             V.SEP, V.PAD}
    out = set(range(0, num_blocks + 1))
    return {"input_tokens": inp, "output_tokens": out, "output_mask": V._make_mask(out)}


VOCAB_INFO = {
    "sort": _sort_vocab_info,
    "python_execution": _python_execution_vocab_info,
    "mini_shrdlu": _mini_shrdlu_vocab_info,
}
