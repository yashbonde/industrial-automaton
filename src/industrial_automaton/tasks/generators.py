import numpy as np
import inspect
from typing import Callable, Dict, List, Optional, Tuple

from ..vocab import PAD, SEP, YIELD, TASK, D, task_idx


def _format_examples(
    inp_lists: List[List[int]],
    out_lists: List[List[int]],
    hard_array_limit: int,
    task_tokens: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format variable-length (input, output) token pairs into padded arrays.

    NSL sequence format:
        [TASK D_t] [input tokens] YIELD [output tokens] EOS

    Loss mask:
        0 for task prefix + input tokens
        1 for YIELD + output tokens + EOS

    Returns:
        inputs:    (N, hard_array_limit) int64
        targets:   (N, hard_array_limit) int64  — shifted left by 1
        loss_mask: (N, hard_array_limit) int64  — 1 on YIELD+output+EOS positions
    """
    if task_tokens is None:
        task_tokens = []

    n = len(inp_lists)
    inp_arrs = np.full((n, hard_array_limit), PAD, dtype=np.int64)
    tgt_arrs = np.full((n, hard_array_limit), PAD, dtype=np.int64)
    mask_arrs = np.zeros((n, hard_array_limit), dtype=np.int64)

    for b, (inp_tokens, out_tokens) in enumerate(zip(inp_lists, out_lists)):
        full_inp = task_tokens + inp_tokens
        full_seq = full_inp + [YIELD] + out_tokens
        if len(full_seq) > hard_array_limit:
            full_seq = full_seq[:hard_array_limit]
        seq_len = len(full_seq)
        full_inp_len = len(full_inp)

        inp_arrs[b, :seq_len] = full_seq
        if seq_len > 1:
            tgt_arrs[b, :seq_len - 1] = full_seq[1:]

        # Mask=1 starts at YIELD position in the TARGET array.
        # tgt[full_inp_len - 1] = full_seq[full_inp_len] = YIELD
        # tgt[full_inp_len] = out_tokens[0], ..., tgt[full_inp_len+len(out)] = EOS
        out_start = min(full_inp_len - 1, hard_array_limit)  # predicting YIELD
        out_end = min(full_inp_len + len(out_tokens), seq_len - 1, hard_array_limit)
        if out_end > out_start:
            mask_arrs[b, out_start:out_end] = 1

    return inp_arrs, tgt_arrs, mask_arrs


def generate_variable_dataset(
    base_task_fn: Callable,
    base_kwargs: Dict,
    num_examples: int,
    max_seqlen_param: int,
    min_seqlen_param: int = 5,
    hard_array_limit: int = 1024,
    task_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset with NSL autoregressive formatting.

    For each example from the task generator:
      1. Get input: list[int], output: list[int]  (variable length, no padding)
      2. Prepend [TASK, D_t] task prefix
      3. Concatenate: [TASK D_t *input YIELD *output EOS]
      4. Build sequence array and target array (shifted left by 1)
      5. Build loss_mask: 0 for task+input, 1 for YIELD+output+EOS
      6. Pad/truncate to hard_array_limit

    Returns:
      inputs:    (N, hard_array_limit) int64  — token IDs
      targets:   (N, hard_array_limit) int64  — next-token targets (shifted left by 1)
      loss_mask: (N, hard_array_limit) int64  — 1 on YIELD+output+EOS positions
    """
    task_tokens = [TASK, D(task_idx(task_name))] if task_name is not None else []

    chunk_size = 64
    num_chunks = int(np.ceil(num_examples / chunk_size))

    # Check if the task function accepts a 'length' or 'rng' parameter.
    # Use the underlying function's signature if it's a partial, then check
    # which args are already bound so we don't double-pass them.
    import functools
    underlying = base_task_fn.func if isinstance(base_task_fn, functools.partial) else base_task_fn
    already_bound = set(base_task_fn.keywords.keys()) if isinstance(base_task_fn, functools.partial) else set()
    sig = inspect.signature(underlying)
    accepts_length = "length" in sig.parameters
    accepts_rng    = "rng" in sig.parameters and "rng" not in already_bound

    if accepts_length:
        print(f"Generating {num_examples} examples with lengths {min_seqlen_param}-{max_seqlen_param}...")
    else:
        print(f"Generating {num_examples} examples (fixed-length task)...")

    rng = np.random.default_rng()

    all_inputs = []
    all_targets = []
    all_masks = []

    # Strip any keys from base_kwargs that are already bound in task_fn (e.g. rng)
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k not in already_bound}

    for _ in range(num_chunks):
        rng_arg = {"rng": rng} if accepts_rng else {}
        if accepts_length:
            # 25% Rule (Neural GPU): Sample shorter sequences occasionally
            if np.random.random() < 0.25 and max_seqlen_param > min_seqlen_param:
                l = np.random.randint(min_seqlen_param, max_seqlen_param)
            else:
                l = np.random.randint(min_seqlen_param, max_seqlen_param + 1)
            data = base_task_fn(batch_size=chunk_size, length=l, **rng_arg, **filtered_kwargs)
        else:
            data = base_task_fn(batch_size=chunk_size, **rng_arg, **filtered_kwargs)

        inp_lists = data["input"]   # list[list[int]]
        out_lists = data["output"]  # list[list[int]]

        chunk_inp, chunk_tgt, chunk_mask = _format_examples(
            inp_lists, out_lists, hard_array_limit, task_tokens
        )
        all_inputs.append(chunk_inp)
        all_targets.append(chunk_tgt)
        all_masks.append(chunk_mask)

    # Concatenate chunks and truncate to num_examples
    inputs = np.concatenate(all_inputs, axis=0)[:num_examples]
    targets = np.concatenate(all_targets, axis=0)[:num_examples]
    loss_mask = np.concatenate(all_masks, axis=0)[:num_examples]

    # Trim to actual maximum sequence length found in this dataset
    has_content = (inputs != PAD) | (targets != PAD) | (loss_mask != 0)
    if has_content.any():
        actual_max_len = np.max(np.where(has_content)[1]) + 1
        inputs = inputs[:, :actual_max_len]
        targets = targets[:, :actual_max_len]
        loss_mask = loss_mask[:, :actual_max_len]

    return inputs, targets, loss_mask


def create_online_batch_generator(
    task_fn: Callable,
    base_kwargs: Dict,
    batch_size: int,
    hard_array_limit: int = 1024,
    task_name: Optional[str] = None,
) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns a callable (length: float) -> (inputs, targets, loss_mask).

    Used with UniformCurriculum for on-the-fly batch generation at random lengths.
    Each call generates `batch_size` examples at exactly `int(length)`.

    Args:
        task_fn: Task generator function (must accept `length` parameter)
        base_kwargs: Extra kwargs to pass to task_fn (e.g., {'rng': rng})
        batch_size: Number of examples per batch
        hard_array_limit: Padded sequence length
        task_name: Task name for TASK prefix (e.g., "parity_check")

    Returns:
        Callable that generates a fresh batch at the given length each call.
    """
    import functools
    task_tokens = [TASK, D(task_idx(task_name))] if task_name is not None else []
    underlying     = task_fn.func if isinstance(task_fn, functools.partial) else task_fn
    already_bound  = set(task_fn.keywords.keys()) if isinstance(task_fn, functools.partial) else set()
    sig = inspect.signature(underlying)
    accepts_length = "length" in sig.parameters
    accepts_rng    = "rng" in sig.parameters and "rng" not in already_bound
    _rng = np.random.default_rng()
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k not in already_bound}

    def generator(length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        l = max(1, int(length))
        rng_arg = {"rng": _rng} if accepts_rng else {}
        if accepts_length:
            data = task_fn(batch_size=batch_size, length=l, **rng_arg, **filtered_kwargs)
        else:
            data = task_fn(batch_size=batch_size, **rng_arg, **filtered_kwargs)
        inp_arrs, tgt_arrs, mask_arrs = _format_examples(
            data["input"], data["output"], hard_array_limit, task_tokens
        )
        
        # Trim to actual maximum sequence length found in this batch
        has_content = (inp_arrs != PAD) | (tgt_arrs != PAD) | (mask_arrs != 0)
        if has_content.any():
            actual_max_len = np.max(np.where(has_content)[1]) + 1
            inp_arrs = inp_arrs[:, :actual_max_len]
            tgt_arrs = tgt_arrs[:, :actual_max_len]
            mask_arrs = mask_arrs[:, :actual_max_len]
            
        return inp_arrs, tgt_arrs, mask_arrs

    return generator


def create_batch_iterator(inputs, targets, loss_mask, batch_size, shuffle=True, seed=None):
    """Create an infinite iterator that yields batches from a dataset.

    Args:
        inputs: Input data array (N, T)
        targets: Target data array (N, T)
        loss_mask: Loss mask array (N, T)
        batch_size: Size of each batch
        shuffle: Whether to shuffle data between epochs (default: True)
        seed: Random seed for shuffling (default: None)

    Yields:
        Tuple of (batch_inputs, batch_targets, batch_loss_mask)
    """
    dataset_size = inputs.shape[0]
    num_batches = dataset_size // batch_size
    shuffle_rng = np.random.default_rng(seed) if shuffle else None

    indices = np.arange(dataset_size)

    while True:
        if shuffle_rng is not None:
            shuffle_rng.shuffle(indices)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            yield (
                inputs[batch_indices],
                targets[batch_indices],
                loss_mask[batch_indices],
            )
