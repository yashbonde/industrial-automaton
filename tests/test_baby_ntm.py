"""Tests for Baby-NTM: smoke tests and D4 training convergence."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import pytest

from industrial_automaton.models.baby_ntm import BabyNTM, BabyNTMState, BabyNTMModelConfig
from industrial_automaton.models.baby_ntm.memory_ops import build_op_matrices, apply_memory_ops
from industrial_automaton.tasks.context_free import generate_dyck_n
from industrial_automaton.trainer import JAXTrainer
from industrial_automaton.config import Settings
from industrial_automaton.vocab import SIZE as VOCAB_SIZE


# --- Memory ops tests ---

class TestMemoryOps:
    def test_rotate_right(self):
        ops = build_op_matrices(5)
        m = jnp.array([[1.], [2.], [3.], [4.], [5.]])
        result = ops[0] @ m
        expected = jnp.array([[5.], [1.], [2.], [3.], [4.]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_rotate_left(self):
        ops = build_op_matrices(5)
        m = jnp.array([[1.], [2.], [3.], [4.], [5.]])
        result = ops[1] @ m
        expected = jnp.array([[2.], [3.], [4.], [5.], [1.]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_no_op(self):
        ops = build_op_matrices(5)
        m = jnp.array([[1.], [2.], [3.], [4.], [5.]])
        result = ops[2] @ m
        np.testing.assert_allclose(result, m, atol=1e-6)

    def test_pop_right(self):
        ops = build_op_matrices(5)
        m = jnp.array([[1.], [2.], [3.], [4.], [5.]])
        result = ops[3] @ m
        expected = jnp.array([[0.], [1.], [2.], [3.], [4.]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_pop_left(self):
        ops = build_op_matrices(5)
        m = jnp.array([[1.], [2.], [3.], [4.], [5.]])
        result = ops[4] @ m
        expected = jnp.array([[2.], [3.], [4.], [5.], [0.]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_apply_memory_ops(self):
        ops = build_op_matrices(5)
        m = jnp.array([[1.], [2.], [3.], [4.], [5.]])
        weights = jnp.array([0., 0., 1., 0., 0.])
        result = apply_memory_ops(m, weights, ops)
        np.testing.assert_allclose(result, m, atol=1e-6)


# --- Model smoke tests ---

class TestBabyNTMSmoke:
    @pytest.fixture
    def model(self):
        config = BabyNTMModelConfig(hidden_size=8, memory_size=10, memory_dim=1)
        return BabyNTM(config=config, key=jax.random.PRNGKey(0))

    def test_forward_shapes(self, model):
        state = model.init_state()
        seq_len = 10
        inputs = jax.nn.one_hot(jnp.ones(seq_len, dtype=jnp.int32), VOCAB_SIZE)
        outputs, final_state = model(inputs, state)
        assert outputs.shape == (seq_len, VOCAB_SIZE)
        assert final_state.memory.shape == (10, 1)
        assert final_state.hidden.shape == (8,)

    def test_no_nan(self, model):
        state = model.init_state()
        inputs = jax.nn.one_hot(jnp.array([1, 3, 5, 2, 4]), VOCAB_SIZE)
        outputs, final_state = model(inputs, state)
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.all(jnp.isfinite(final_state.memory))

    def test_jit_compiles(self, model):
        state = model.init_state()
        inputs = jax.nn.one_hot(jnp.ones(5, dtype=jnp.int32), VOCAB_SIZE)

        @jax.jit
        def run(m, inp, s):
            return m(inp, s)

        outputs, _ = run(model, inputs, state)
        assert outputs.shape == (5, VOCAB_SIZE)

    def test_gradients_nonzero(self, model):
        state = model.init_state()
        inputs = jax.nn.one_hot(jnp.array([1, 2, 3]), VOCAB_SIZE)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            outputs, _ = m(inputs, state)
            return jnp.mean(outputs ** 2)

        loss, grads = loss_fn(model)
        flat_grads = jax.tree.leaves(grads)
        any_nonzero = any(jnp.any(g != 0) for g in flat_grads if eqx.is_array(g))
        assert any_nonzero


# --- D4 training convergence test ---

class TestD4Convergence:
    def test_loss_decreases(self, tmp_path):
        key = jax.random.PRNGKey(42)
        config = BabyNTMModelConfig(hidden_size=12, memory_size=25, memory_dim=1)
        model = BabyNTM(config=config, key=key)

        data = generate_dyck_n(batch_size=64, length=20, n=4, rng=np.random.default_rng(0))
        inputs_np = data["input"]
        labels_np = data["output"]

        batch = (
            jax.nn.one_hot(jnp.array(inputs_np), VOCAB_SIZE),
            jnp.array(labels_np, dtype=jnp.float32),
        )

        def loss_fn(model, batch, key):
            inputs, labels = batch
            def single(inp, lab):
                state = model.init_state()
                outputs, _ = model(inp, state)
                pred = outputs[-1, 0]
                return -lab * jnp.log(pred + 1e-7) - (1 - lab) * jnp.log(1 - pred + 1e-7)
            loss = jnp.mean(jax.vmap(single)(inputs, labels))
            return loss, {}

        settings = Settings(
            tr_run_name="test_d4",
            tr_save_folder=str(tmp_path),
            tr_max_steps=200,
            tr_logging_steps=50,
            tr_save_steps=10000,
            tr_eval_steps=10000,
            tr_lr=1e-3,
            tr_precision="fp32",
            _cli_parse_args=[],
        )

        trainer = JAXTrainer(model, loss_fn, settings)

        def data_gen():
            while True:
                yield batch

        history = trainer.fit(data_gen())

        initial_loss = history[0].loss
        final_loss = history[-1].loss

        assert final_loss < 0.8 * initial_loss, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )


if __name__ == "__main__":
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        tester = TestD4Convergence()
        tester.test_loss_decreases(pathlib.Path(d))
        print("D4 convergence test passed!")
