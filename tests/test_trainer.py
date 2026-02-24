"""End-to-end tests for JAXTrainer: training, logging, checkpointing, eval, rotation."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from industrial_automaton.config import Settings
from industrial_automaton.trainer import JAXTrainer, StepMetrics, build_optimizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class LinearModel(eqx.Module):
    weights: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, key):
        wkey, bkey = jax.random.split(key)
        self.weights = jax.random.normal(wkey, (1,))
        self.bias = jax.random.normal(bkey, (1,))

    def __call__(self, x):
        return x * self.weights + self.bias


def loss_fn(model, batch, key):
    x, y = batch
    preds = jax.vmap(model)(x)
    loss = jnp.mean((preds - y) ** 2)
    return loss, {"accuracy": -loss}


def _make_batch(curriculum_bound=1.0):
    x = jnp.linspace(0, curriculum_bound, 32).reshape(-1, 1)
    y = 2.0 * x + 0.5
    return x, y


def _iter_batches():
    while True:
        yield _make_batch()


@pytest.fixture
def base_settings(tmp_path):
    """Minimal settings pointing at tmp_path, short run for speed."""
    return dict(
        tr_save_folder=str(tmp_path),
        tr_lr=1e-2,
        tr_precision="fp32",
    )


def _run_dir(tmp_path, name):
    return tmp_path / name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunDirectorySetup:
    """Verify that __init__ creates the full run directory tree + config."""

    def test_dirs_and_config_created(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="init_test",
            tr_max_steps=1,
            **base_settings
        )
        JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)

        rd = _run_dir(tmp_path, "init_test")
        assert rd.is_dir()
        assert (rd / "ckpt").is_dir()
        assert (rd / "tb").is_dir()
        assert (rd / "config.json").is_file()

        cfg = json.loads((rd / "config.json").read_text())
        assert cfg["tr_run_name"] == "init_test"
        assert cfg["tr_lr"] == 1e-2


class TestLogFile:
    """Verify dual logging (stdout captured by capsys, file on disk)."""

    def test_log_file_written(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="log_test",
            tr_max_steps=10,
            tr_logging_steps=5,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        trainer.fit(_iter_batches())

        log_path = _run_dir(tmp_path, "log_test") / "logs"
        assert log_path.is_file()
        text = log_path.read_text()
        # Should contain the "Starting training" line + at least one step log
        assert "Starting training" in text
        assert "Step 00005" in text
        assert "Tokens:" in text
        assert "Training complete" in text


class TestCheckpointing:
    """Save at tr_save_steps, verify state.eqx files exist."""

    def test_checkpoint_files_created(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="ckpt_test",
            tr_max_steps=10,
            tr_save_steps=5,
            tr_logging_steps=100,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        trainer.fit(_iter_batches())

        ckpt = _run_dir(tmp_path, "ckpt_test") / "ckpt"
        assert (ckpt / "step-5" / "state.eqx").is_file()
        assert (ckpt / "step-10" / "state.eqx").is_file()


class TestCheckpointRotation:
    """Only tr_save_limit most recent checkpoints kept (+ best if outside window)."""

    def test_rotation_keeps_limit(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="rot_test",
            tr_max_steps=20,
            tr_save_steps=5,
            tr_save_limit=2,
            tr_logging_steps=100,
            tr_eval_steps=100,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        trainer.fit(_iter_batches())

        ckpt = _run_dir(tmp_path, "rot_test") / "ckpt"
        step_dirs = [d for d in ckpt.iterdir() if d.is_dir() and d.name.startswith("step-")]
        # At most tr_save_limit (2) dirs remain
        assert len(step_dirs) <= 2
        # The two most recent (step-15, step-20) should survive
        names = {d.name for d in step_dirs}
        assert "step-20" in names


class TestBestSymlink:
    """eval_fn triggers best symlink creation and update."""

    def test_best_symlink_points_to_best_eval(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="best_test",
            tr_max_steps=10,
            tr_save_steps=5,
            tr_eval_steps=5,
            tr_logging_steps=100,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)

        # eval_fn always returns a loss so best gets set
        def eval_fn(model):
            return StepMetrics(loss=jnp.array(0.5))

        trainer.fit(_iter_batches(), eval_fn=eval_fn)

        best = _run_dir(tmp_path, "best_test") / "ckpt" / "best"
        assert best.is_symlink()
        # Should point to a real step dir with state.eqx
        assert (best / "state.eqx").is_file()


class TestEvalJsonl:
    """eval_fn writes structured JSONL."""

    def test_eval_logs_written(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="eval_test",
            tr_max_steps=10,
            tr_eval_steps=5,
            tr_save_steps=100,
            tr_logging_steps=100,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)

        call_count = 0

        def eval_fn(model):
            nonlocal call_count
            call_count += 1
            return StepMetrics(loss=jnp.array(1.0 / call_count), accuracy=jnp.array(0.9))

        trainer.fit(_iter_batches(), eval_fn=eval_fn)

        jsonl_path = _run_dir(tmp_path, "eval_test") / "eval_logs.jsonl"
        assert jsonl_path.is_file()
        lines = [json.loads(l) for l in jsonl_path.read_text().strip().splitlines()]
        assert len(lines) == 2  # step 5 and step 10
        assert lines[0]["step"] == 5
        assert lines[1]["step"] == 10
        assert "loss" in lines[0]
        assert "accuracy" in lines[0]
        assert "timestamp" in lines[0]


class TestGradientAccumulation:
    """With accum_steps=N, each logical step consumes N micro-batches."""

    def test_accum_produces_same_step_count(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="accum_test", tr_max_steps=4,
            tr_gradient_accumulation_steps=3, tr_logging_steps=100,
            tr_save_steps=100, **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        history = trainer.fit(_iter_batches())
        assert len(history) == 4
        assert int(trainer.state.step) == 4


class TestTokenCounting:
    """num_input_tokens_seen accumulates batch_size * seq_len."""

    def test_token_count(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="token_test",
            tr_max_steps=3,
            tr_logging_steps=100,
            tr_save_steps=100,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        # batch shape: (32, 1) → 32 tokens per step
        trainer.fit(_iter_batches())
        assert trainer.num_input_tokens_seen == 3 * 32


class TestCurriculumUpdate:
    """curriculum_fn modifies curriculum_bound during training."""

    def test_curriculum_increases(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="curr_test",
            tr_max_steps=50,
            tr_logging_steps=100,
            tr_save_steps=100,
            **base_settings,
        )
        model = LinearModel(jax.random.PRNGKey(42))
        trainer = JAXTrainer(model, loss_fn, settings)

        initial = float(trainer.state.curriculum_bound)
        trainer.fit(
            data_generator=lambda cb: _make_batch(cb),
            curriculum_fn=lambda b, m: b + 1.0 if m.loss < 0.01 else b,
        )
        assert float(trainer.state.curriculum_bound) >= initial


class TestBuildOptimizer:
    """build_optimizer respects scheduler and optimizer selection."""

    def test_cosine_schedule(self):
        s = Settings(
            tr_lr_scheduler="cosine",
            tr_warmup_steps=10,
            tr_max_steps=100
        )
        opt = build_optimizer(s)
        assert opt is not None

    def test_linear_schedule(self):
        s = Settings(tr_lr_scheduler="linear", tr_max_steps=100)
        opt = build_optimizer(s)
        assert opt is not None

    def test_unknown_optimizer_raises(self):
        s = Settings(tr_optimizer="nonesuch")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(s)

    def test_unknown_scheduler_raises(self):
        s = Settings(tr_lr_scheduler="nonesuch")
        with pytest.raises(ValueError, match="Unknown lr scheduler"):
            build_optimizer(s)


class TestTensorBoard:
    """TensorBoard writer creates event files when enabled."""

    def test_tb_event_files(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="tb_test",
            tr_max_steps=5,
            tr_logging_steps=5,
            tr_save_steps=100,
            tr_tensorboard=True,
            **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        trainer.fit(_iter_batches())

        tb_dir = _run_dir(tmp_path, "tb_test") / "tb"
        # tensorboardX writes events.out.tfevents.* files
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_tb_symlink(self, tmp_path, base_settings):
        link_target = tmp_path / "external_tb"
        settings = Settings(
            tr_run_name="tb_link_test",
            tr_max_steps=1,
            tr_logging_steps=1,
            tr_save_steps=100,
            tr_tensorboard=True,
            tr_tensorboard_log_dir=str(link_target),
            **base_settings,
        )
        JAXTrainer(LinearModel(jax.random.PRNGKey(0)), loss_fn, settings)
        assert link_target.is_symlink()


class TestSaveLoad:
    """save() / load() round-trips the state."""

    def test_save_load_roundtrip(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="sl_test", tr_max_steps=5, tr_logging_steps=100,
            tr_save_steps=100, **base_settings,
        )
        model = LinearModel(jax.random.PRNGKey(0))
        trainer = JAXTrainer(model, loss_fn, settings)
        trainer.fit(_iter_batches())

        save_path = str(tmp_path / "manual_save.eqx")
        trainer.save(save_path)

        # Create a fresh trainer with same model structure
        settings2 = Settings(
            tr_run_name="sl_test2", tr_max_steps=1, tr_logging_steps=100,
            tr_save_steps=100, **base_settings,
        )
        trainer2 = JAXTrainer(LinearModel(jax.random.PRNGKey(99)), loss_fn, settings2)
        trainer2.load(save_path)

        # After load, weights should match
        w1 = trainer.state.model.weights
        w2 = trainer2.state.model.weights
        assert jnp.allclose(w1, w2, atol=1e-6)


class TestLossDecreases:
    """Sanity: loss goes down on a trivial regression."""

    def test_loss_decreases(self, tmp_path, base_settings):
        settings = Settings(
            tr_run_name="loss_test", tr_max_steps=100, tr_logging_steps=100,
            tr_save_steps=1000, **base_settings,
        )
        trainer = JAXTrainer(LinearModel(jax.random.PRNGKey(42)), loss_fn, settings)
        history = trainer.fit(_iter_batches())
        assert float(history[-1].loss) < float(history[0].loss)
