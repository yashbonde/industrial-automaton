"""Baby-NTM memory operations: 5 deterministic permutation/shift matrices."""

import jax.numpy as jnp


def build_op_matrices(n: int) -> jnp.ndarray:
    """Build (5, N, N) stack of operation matrices for memory size N.

    Operations: [rotate_right, rotate_left, no_op, pop_right, pop_left]
    """
    I = jnp.eye(n)

    rotate_right = jnp.roll(I, 1, axis=0)
    rotate_left = jnp.roll(I, -1, axis=0)
    no_op = I

    # pop_right: shift rows down, zero-fill top row
    # Row i gets row i-1's content; row 0 becomes zero
    pop_right = jnp.zeros((n, n))
    if n > 1:
        pop_right = pop_right.at[1:, :-1].set(jnp.eye(n - 1))

    # pop_left: shift rows up, zero-fill bottom row
    # Row i gets row i+1's content; row N-1 becomes zero
    pop_left = jnp.zeros((n, n))
    if n > 1:
        pop_left = pop_left.at[:-1, 1:].set(jnp.eye(n - 1))

    return jnp.stack([rotate_right, rotate_left, no_op, pop_right, pop_left])


def apply_memory_ops(
    memory: jnp.ndarray,
    action_weights: jnp.ndarray,
    op_matrices: jnp.ndarray,
) -> jnp.ndarray:
    """Apply weighted combination of memory operations.

    Args:
        memory: (N, M) memory matrix
        action_weights: (5,) softmax weights over operations
        op_matrices: (5, N, N) operation matrices

    Returns:
        (N, M) new memory matrix: sum_i a[i] * OP[i] @ M
    """
    # (5, N, N) @ (N, M) -> (5, N, M), then weight by (5,) and sum
    return jnp.einsum("i,ijk,kl->jl", action_weights, op_matrices, memory)
