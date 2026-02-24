"""Tests for Suzgun Stack-RNN."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest
from industrial_automaton.models.suzgun_stack_rnn import SuzgunStackRNN, StackRNNState, SuzgunStackRNNConfig
from industrial_automaton.trainer import JAXTrainer
from industrial_automaton.config import Settings
from industrial_automaton.vocab import SIZE as VOCAB_SIZE

# --- Model smoke tests ---

class TestSuzgunStackRNNSmoke:
    @pytest.fixture
    def model(self):
        config = SuzgunStackRNNConfig(
            hidden_size=8,
            stack_depth=10,
            value_dim=1,
        )
        return SuzgunStackRNN(config=config, key=jax.random.PRNGKey(0))

    def test_forward_shapes(self, model):
        state = model.init_state()
        seq_len = 10
        inputs = jax.nn.one_hot(jnp.ones(seq_len, dtype=jnp.int32), VOCAB_SIZE)
        outputs, final_state = model(inputs, state)
        assert outputs.shape == (seq_len, VOCAB_SIZE)
        assert final_state.stack.shape == (10, 1)
        assert final_state.hidden.shape == (8,)

    def test_no_nan(self, model):
        state = model.init_state()
        inputs = jax.nn.one_hot(jnp.array([1, 2, 1, 2]), VOCAB_SIZE)
        outputs, final_state = model(inputs, state)
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.all(jnp.isfinite(final_state.stack))

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
        inputs = jax.nn.one_hot(jnp.array([1, 2]), VOCAB_SIZE)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            outputs, _ = m(inputs, state)
            return jnp.mean(outputs ** 2)

        loss, grads = loss_fn(model)
        flat_grads = jax.tree.leaves(grads)
        any_nonzero = any(jnp.any(g != 0) for g in flat_grads if eqx.is_array(g))
        assert any_nonzero

# --- Convergence test on a^n b^n ---

def generate_anbn_data(batch_size, max_n, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    
    length = 2 * max_n
    inputs = np.zeros((batch_size, length), dtype=np.int32)
    labels = np.zeros(batch_size, dtype=np.int32)
    
    for b in range(batch_size):
        if rng.random() < 0.5:
            # Valid a^n b^n
            n = rng.integers(1, max_n + 1)
            seq = [1]*n + [2]*n
            inputs[b, :len(seq)] = seq
            labels[b] = 1
        else:
            # Invalid: mixed or wrong counts
            case = rng.integers(0, 3)
            if case == 0: # Wrong counts: a^n b^m
                n = rng.integers(1, max_n + 1)
                m = rng.integers(1, max_n + 1)
                while m == n:
                    m = rng.integers(1, max_n + 1)
                seq = [1]*n + [2]*m
            elif case == 1: # Interleaved: (ab)^n
                n = rng.integers(1, max_n + 1)
                seq = [1, 2] * n
            else: # Wrong order: b^n a^n
                n = rng.integers(1, max_n + 1)
                seq = [2]*n + [1]*n
            
            # Clip to length
            if len(seq) > length:
                seq = seq[:length]
            inputs[b, :len(seq)] = seq
            labels[b] = 0
            
    return inputs, labels

class TestAnBnConvergence:
    def test_loss_decreases(self, tmp_path):
        key = jax.random.PRNGKey(42)
        config = SuzgunStackRNNConfig(
            hidden_size=16,
            stack_depth=20,
            value_dim=1,
        )
        model = SuzgunStackRNN(config=config, key=key)

        max_n = 5
        inputs_np, labels_np = generate_anbn_data(batch_size=128, max_n=max_n)
        
        batch = (
            jax.nn.one_hot(jnp.array(inputs_np), VOCAB_SIZE),
            jnp.array(labels_np, dtype=jnp.float32),
        )

        def loss_fn(model, batch, key):
            inputs, labels = batch
            def single(inp, lab):
                state = model.init_state()
                outputs, _ = model(inp, state)
                logits = outputs[-1]
                pred_logit = logits[0]
                pred = jax.nn.sigmoid(pred_logit)
                return -lab * jnp.log(pred + 1e-7) - (1 - lab) * jnp.log(1 - pred + 1e-7)
            
            loss = jnp.mean(jax.vmap(single)(inputs, labels))
            return loss, {}

        settings = Settings(
            tr_run_name="test_anbn",
            tr_save_folder=str(tmp_path),
            tr_max_steps=300,
            tr_logging_steps=50,
            tr_save_steps=10000,
            tr_eval_steps=10000,
            tr_lr=5e-3,
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

        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        assert final_loss < initial_loss, "Loss did not decrease"
        assert final_loss < 0.6, f"Final loss too high: {final_loss:.4f}"

if __name__ == "__main__":
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        tester = TestAnBnConvergence()
        tester.test_loss_decreases(pathlib.Path(d))
