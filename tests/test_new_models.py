"""Tests for Tape-RNN, Transformer, and LSTM."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest
from industrial_automaton.models.tape_rnn import TapeRNN, TapeRNNConfig
from industrial_automaton.models.transformer import Transformer, TransformerConfig
from industrial_automaton.models.lstm import LSTM, LSTMConfig
from industrial_automaton.trainer import JAXTrainer
from industrial_automaton.config import Settings
from industrial_automaton.vocab import SIZE as VOCAB_SIZE

def generate_parity_data(batch_size, length, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    inputs = rng.integers(0, 2, size=(batch_size, length))
    labels = np.sum(inputs, axis=1) % 2
    # vocab: 0, 1 -> one_hot(2)
    return inputs, labels

def parity_loss_fn(model, batch, key):
    inputs_idx, labels = batch
    inputs = jax.nn.one_hot(inputs_idx, VOCAB_SIZE)
    def single(inp, lab):
        state = model.init_state()
        outputs, _ = model(inp, state)
        logits = outputs[-1]
        # Binary classification on index 1
        pred = jax.nn.sigmoid(logits[1] - logits[0])
        return -lab * jnp.log(pred + 1e-7) - (1 - lab) * jnp.log(1 - pred + 1e-7)
    
    loss = jnp.mean(jax.vmap(single)(inputs, labels))
    return loss, {}

@pytest.mark.parametrize("model_class", [TapeRNN, Transformer, LSTM])
def test_convergence_parity(model_class, tmp_path):
    key = jax.random.PRNGKey(0)
    if model_class == Transformer:
        config = TransformerConfig(embed_dim=16, num_heads=2, num_layers=1, max_seq_len=20)
    elif model_class == TapeRNN:
        config = TapeRNNConfig(hidden_size=16, memory_size=10)
    else:
        config = LSTMConfig(hidden_size=16)
    
    model = model_class(config=config, key=key)

    inputs_np, labels_np = generate_parity_data(64, 10)
    batch = (jnp.array(inputs_np), jnp.array(labels_np, dtype=jnp.float32))

    settings = Settings(
        tr_run_name=f"test_{model_class.__name__.lower()}",
        tr_save_folder=str(tmp_path),
        tr_max_steps=200,
        tr_logging_steps=50,
        tr_lr=1e-3,
        tr_precision="fp32",
        _cli_parse_args=[],
    )

    trainer = JAXTrainer(model, parity_loss_fn, settings)
    def data_gen():
        while True: yield batch

    history = trainer.fit(data_gen())
    assert history[-1].loss < history[0].loss
