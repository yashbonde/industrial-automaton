import json
import time
import numpy as np
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from industrial_automaton.config import Settings
from industrial_automaton.vocab import PAD, YIELD as YIELD_TOKEN
from industrial_automaton.curriculum import init_curriculum_state


class TrainingDivergedError(Exception):
    """Raised when training diverges (NaN, gradient explosion, etc.)."""
    pass


class DivergenceMonitor:
    """Monitors training health across multiple signals.

    Detects:
    - NaN/Inf loss (immediate failure)
    - Gradient explosion (L2 norm > threshold)
    - Gradient vanishing (L2 norm < 1e-6 for N consecutive steps)
    - Loss plateau (moving avg unchanged for N steps)
    """

    def __init__(
        self,
        grad_explosion_threshold: float = 1000.0,  # Pre-clip norm; optimizer clips to 1.0 anyway
        vanishing_patience: int = 100,
        plateau_patience: int = 200,
        plateau_threshold: float = 1e-6,
        ema_decay: float = 0.9,
    ):
        self.grad_explosion_threshold = grad_explosion_threshold
        self.vanishing_patience = vanishing_patience
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.ema_decay = ema_decay

        # State tracking
        self.vanishing_streak = 0
        self.loss_ema = None
        self.plateau_count = 0

    def check(self, loss: float, grads, step: int):
        """Check for divergence signals.

        Args:
            loss: Current loss value
            grads: Gradient tree
            step: Current training step

        Raises:
            TrainingDivergedError: If divergence is detected
        """
        # 1. NaN/Inf: immediate failure
        if np.isnan(loss) or np.isinf(loss):
            raise TrainingDivergedError(f"Loss NaN/Inf at step {step}: {loss}")

        # 2. Gradient norm explosion
        grad_norm = float(optax.global_norm(grads))
        if grad_norm > self.grad_explosion_threshold:
            raise TrainingDivergedError(
                f"Gradient explosion: norm={grad_norm:.2f} at step {step}"
            )

        # 3. Gradient vanishing (track consecutive steps)
        if grad_norm < 1e-6:
            self.vanishing_streak += 1
            if self.vanishing_streak > self.vanishing_patience:
                raise TrainingDivergedError(
                    f"Gradient vanishing for {self.vanishing_streak} steps at step {step}"
                )
        else:
            self.vanishing_streak = 0

        # 4. Loss plateau (EMA-based)
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            old_ema = self.loss_ema
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * loss

            # Check if loss has plateaued
            if abs(self.loss_ema - old_ema) < self.plateau_threshold:
                self.plateau_count += 1
                if self.plateau_count > self.plateau_patience:
                    raise TrainingDivergedError(
                        f"Loss plateaued at {self.loss_ema:.4f} for {self.plateau_count} steps at step {step}"
                    )
            else:
                self.plateau_count = 0


class TrainState(eqx.Module):
    """Unified state for the trainer, fully compatible with JAX transformations."""
    model: eqx.Module
    opt_state: optax.OptState
    key: jax.Array
    step: jnp.ndarray
    curriculum_state: Optional[Any] = None  # CurriculumState or None for backward compatibility
    curriculum_bound: jnp.ndarray = None  # Legacy field for backward compatibility


class StepMetrics(NamedTuple):
    """Container for metrics from a single training step."""
    loss: jnp.ndarray
    accuracy: Optional[jnp.ndarray] = None
    aux: Optional[Dict[str, jnp.ndarray]] = None


def build_optimizer(settings: Settings) -> optax.GradientTransformation:
    """Build an optax optimizer chain from settings."""
    # Use constant learning rate (no scheduler)
    lr = settings.learning_rate

    # Optimizer
    opt_name = settings.optimizer.lower()
    opt_map = {"adam": optax.adam, "adamw": optax.adamw, "sgd": optax.sgd}
    if opt_name not in opt_map:
        raise ValueError(f"Unknown optimizer: {opt_name}. Choose from {list(opt_map)}")
    opt_kwargs = settings.optimizer_kwargs or {}
    optimizer = opt_map[opt_name](learning_rate=lr, **opt_kwargs)

    # Gradient clipping with max_grad_norm=1.0
    return optax.chain(optax.clip_by_global_norm(1.0), optimizer)


def _cast_bf16(model):
    """Cast all arrays in model to bfloat16."""
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) else x, model
    )


def _cast_fp32(tree):
    """Cast all arrays in tree to float32."""
    return jax.tree.map(
        lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, tree
    )


class Trainer:
    """Feature-rich JAX trainer driven by Settings config.

    Features: checkpoint saving with best-symlink and rotation, TensorBoard,
    file+CLI logging, eval JSONL logging, LR scheduler, gradient accumulation,
    gradient clipping, and mixed precision.
    """
    def __init__(
        self,
        model: eqx.Module,
        settings: Settings,
        task_metadata: Optional[Dict[str, Any]] = None,
        curriculum: Optional[Any] = None,  # CurriculumStrategy instance
        enable_divergence_monitor: bool = True,
        eval_inputs: Optional[Any] = None,
        eval_labels: Optional[Any] = None,
        eval_loss_mask: Optional[Any] = None,
        eval_dataset_size: Optional[int] = None,
    ):
        self.settings = settings
        self.task_metadata = task_metadata
        self.curriculum = curriculum
        self.num_input_tokens_seen = 0
        self.num_output_tokens_produced = 0
        self.unique_input_tokens = set()

        self.eval_inputs = eval_inputs
        self.eval_labels = eval_labels
        self.eval_loss_mask = eval_loss_mask
        self.eval_dataset_size = eval_dataset_size

        # Divergence monitoring
        self.divergence_monitor = DivergenceMonitor() if enable_divergence_monitor else None
        self.early_stopping_counter = 0
        self.best_eval_accuracy = -float('inf')
        self.no_improvement_counter = 0
        self.early_stopping_patience = getattr(settings, 'early_stopping_patience', 20)

        # Build optimizer
        self.optimizer = build_optimizer(settings)

        # Init state
        key = jax.random.PRNGKey(settings.seed)
        _, state_key = jax.random.split(key)

        # Handle pure bf16 training
        if settings.precision == "bf16":
            model = _cast_bf16(model)

        # Initialize curriculum state if curriculum provided
        curriculum_state = None
        if curriculum:
            curriculum_state = init_curriculum_state(
                strategy=curriculum,
                min_bound=10,  # Sensible minimum
                max_bound=settings.max_seqlen,
                initial_bound=10,  # Start small
            )

        self.state = TrainState(
            model=model,
            opt_state=self.optimizer.init(eqx.filter(model, eqx.is_array)),
            key=state_key,
            step=jnp.array(0, dtype=jnp.int32),
            curriculum_state=curriculum_state,
            curriculum_bound=jnp.array(1.0, dtype=jnp.float32),  # Legacy field
        )

        # Run directory
        self.run_dir = Path(settings.save_folder) / settings.run_name
        self.ckpt_dir = self.run_dir / "ckpt"
        self.tb_dir = self.run_dir / "tb"
        self.log_file = self.run_dir / "logs"
        self.eval_log_file = self.run_dir / "eval_logs.jsonl"
        self.config_file = self.run_dir / "config.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.tb_dir.mkdir(exist_ok=True)

        # Save config
        self.config_file.write_text(settings.model_dump_json(indent=2))

        # TensorBoard - DISABLED
        # self.tb_writer = None
        # if settings.tr_tensorboard:
        #     try:
        #         from tensorboardX import SummaryWriter
        #         self.tb_writer = SummaryWriter(str(self.tb_dir))
        #     except ImportError:
        #         self._log("WARNING: tensorboardX not installed, disabling TensorBoard")
        #
        #     if settings.tr_tensorboard_log_dir:
        #         tb_link = Path(settings.tr_tensorboard_log_dir)
        #         tb_link.parent.mkdir(parents=True, exist_ok=True)
        #         if tb_link.is_symlink() or tb_link.exists():
        #             tb_link.unlink()
        #         tb_link.symlink_to(self.tb_dir.resolve())

        # JIT compile steps
        self._jit_compute_grads = eqx.filter_jit(self._compute_grads)
        self._jit_apply_grads = eqx.filter_jit(self._apply_grads)

    def _log(self, msg: str):
        """Print to stdout and append to log file."""
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def _compute_grads(self, model, batch, key):
        """Compute loss, aux, and gradients for a single micro-batch."""
        precision = self.settings.precision

        if precision == "mixed-bf16-fp32":
            model_compute = _cast_bf16(model)
        else:
            # For "bf16" or "fp32", model is already in correct precision from __init__
            model_compute = model

        @eqx.filter_value_and_grad(has_aux=True)
        def compute(m, batch, key):
            loss, aux = loss_fn(m, batch, key, task_metadata=self.task_metadata)
            return loss, aux

        (loss, aux), grads = compute(model_compute, batch, key)

        if precision == "mixed-bf16-fp32":
            grads = _cast_fp32(grads)

        # In pure "bf16" mode, grads stay in bf16 and are applied to bf16 weights.

        return loss, aux, grads

    def _apply_grads(self, state: TrainState, grads) -> TrainState:
        """Apply averaged gradients to state."""
        updates, new_opt_state = self.optimizer.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)
        return TrainState(
            model=new_model,
            opt_state=new_opt_state,
            key=state.key,
            step=state.step + 1,
            curriculum_state=state.curriculum_state,
            curriculum_bound=state.curriculum_bound,
        )

    def _save_checkpoint(self, step: int):
        """Save checkpoint and rotate old ones."""
        step_dir = self.ckpt_dir / f"step-{step}"
        step_dir.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(str(step_dir / "state.eqx"), self.state)

        # Rotate: find all step dirs, keep only save_limit most recent
        best_link = self.ckpt_dir / "best"
        best_target = None
        if best_link.is_symlink():
            best_target = best_link.resolve()

        step_dirs = sorted(
            [d for d in self.ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
            key=lambda d: int(d.name.split("-")[1]),
        )

        if len(step_dirs) > self.settings.save_limit:
            to_remove = step_dirs[: len(step_dirs) - self.settings.save_limit]
            for d in to_remove:
                if best_target and d.resolve() == best_target:
                    continue
                import shutil
                shutil.rmtree(d)

    def _update_best(self, step: int):
        """Update best symlink to point to given step."""
        best_link = self.ckpt_dir / "best"
        target = self.ckpt_dir / f"step-{step}"
        if best_link.is_symlink() or best_link.exists():
            best_link.unlink()
        best_link.symlink_to(target.resolve())

    def _log_eval(self, step: int, metrics: StepMetrics):
        """Append eval metrics to JSONL file."""
        record = {
            "step": step,
            "loss": float(metrics.loss),
            "timestamp": time.time(),
        }
        if metrics.accuracy is not None:
            record["accuracy"] = float(metrics.accuracy)
        with open(self.eval_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def update_curriculum(self, new_bound: float):
        """Updates the curriculum bound in the current state."""
        self.state = eqx.tree_at(
            lambda s: s.curriculum_bound, self.state, jnp.array(new_bound)
        )

    def fit(
        self,
        data_generator: Union[Iterator, Callable[[float], Any]],
        curriculum_fn: Optional[Callable[[float, StepMetrics], float]] = None,
    ) -> List[StepMetrics]:
        """Training loop.

        Args:
            data_generator: Iterator yielding batches, or callable(curriculum_bound) -> batch.
            eval_fn: Optional callable(model) -> StepMetrics for evaluation.
            curriculum_fn: Optional callable(current_bound, metrics) -> new_bound.
        """
        settings = self.settings
        num_steps = settings.max_steps
        accum_steps = 1  # No gradient accumulation
        history = []
        start_time = time.time()
        is_iterator = isinstance(data_generator, Iterator)
        best_eval_loss = float("inf")
        final_eval_accuracy = 0.0

        self._log(f"Starting training for {num_steps} steps")

        for step_idx in range(num_steps):
            # Split key for this step
            self.state = eqx.tree_at(
                lambda s: s.key,
                self.state,
                jax.random.fold_in(self.state.key, step_idx),
            )

            total_loss = 0.0
            total_aux = {}
            accumulated_grads = None

            for micro in range(accum_steps):
                if is_iterator:
                    batch = next(data_generator)
                else:
                    # Use functional curriculum state bound if available, else legacy field
                    if self.state.curriculum_state is not None:
                        curr_bound = float(self.state.curriculum_state.current_bound)
                    else:
                        curr_bound = float(self.state.curriculum_bound)
                    batch = data_generator(curr_bound)

                # Token counting (input and output)
                if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                    input_batch = batch[0]
                    loss_mask_batch = batch[2]

                    if hasattr(input_batch, "shape") and len(input_batch.shape) >= 2:
                        self.num_input_tokens_seen += int(input_batch.shape[0] * input_batch.shape[1])
                        unique_in_batch = set(int(x) for x in jnp.unique(input_batch).tolist())
                        self.unique_input_tokens.update(unique_in_batch)

                    if hasattr(loss_mask_batch, "shape"):
                        self.num_output_tokens_produced += int(np.sum(loss_mask_batch))

                key = jax.random.fold_in(self.state.key, micro)
                loss, aux, grads = self._jit_compute_grads(self.state.model, batch, key)

                total_loss += float(loss)
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree.map(
                        lambda a, b: a + b, accumulated_grads, grads
                    )

                if aux:
                    for k, v in aux.items():
                        total_aux[k] = total_aux.get(k, 0.0) + float(v)

            # Average grads
            avg_grads = jax.tree.map(lambda g: g / accum_steps, accumulated_grads)
            avg_loss = total_loss / accum_steps
            avg_aux = {k: v / accum_steps for k, v in total_aux.items()}

            # Check for divergence
            if self.divergence_monitor:
                self.divergence_monitor.check(avg_loss, avg_grads, step_idx)

            # Apply
            self.state = self._jit_apply_grads(self.state, avg_grads)

            accuracy = avg_aux.get("accuracy")
            acc_val = jnp.array(accuracy) if accuracy is not None else None
            metrics = StepMetrics(loss=jnp.array(avg_loss), accuracy=acc_val, aux=avg_aux or None)
            history.append(metrics)

            current_step = int(self.state.step)

            # Timeout check
            if settings.timeout is not None and (time.time() - start_time) > settings.timeout:
                self._log(f"Timeout ({settings.timeout}s) reached at step {current_step}. Stopping.")
                break

            # Curriculum - new functional system
            if self.curriculum and self.state.curriculum_state:
                metrics_dict = {"loss": float(avg_loss), "accuracy": float(accuracy) if accuracy is not None else 0.0}
                new_curr_state = self.curriculum.update(self.state.curriculum_state, metrics_dict)
                self.state = eqx.tree_at(
                    lambda s: s.curriculum_state,
                    self.state,
                    new_curr_state
                )

            # Legacy curriculum support
            if curriculum_fn:
                new_bound = curriculum_fn(float(self.state.curriculum_bound), metrics)
                if new_bound != float(self.state.curriculum_bound):
                    self.update_curriculum(new_bound)

            # Logging (every 10 steps)
            if current_step % 10 == 0:
                elapsed = time.time() - start_time
                token_acc = avg_aux.get("token_accuracy")
                seq_acc = avg_aux.get("sequence_accuracy")
                acc_str = ""
                if token_acc is not None:
                    acc_str = f" | TokAcc: {token_acc:.4f}"
                if seq_acc is not None:
                    acc_str += f" | SeqAcc: {seq_acc:.4f}"

                # Show curriculum progress (new system or legacy)
                if self.state.curriculum_state:
                    curr_str = f" | Curr: {self.state.curriculum_state.current_bound}"
                else:
                    curr_str = f" | Curr: {self.state.curriculum_bound:.2f}"

                # Format token counts
                def format_tokens(count):
                    if count >= 1_000_000:
                        return f"{count / 1_000_000:.2f}mn"
                    elif count >= 1_000:
                        return f"{count / 1_000:.0f}k"
                    else:
                        return str(count)

                tokens_in_fmt = format_tokens(self.num_input_tokens_seen)
                tokens_out_fmt = format_tokens(self.num_output_tokens_produced)

                msg = (
                    f"Step {current_step:05d} | Loss: {avg_loss:.4f}{acc_str}{curr_str}"
                    f" | Tokens In: {tokens_in_fmt} | Tokens Out: {tokens_out_fmt} | Time: {elapsed:.2f}s"
                )
                self._log(msg)

                # TensorBoard logging - DISABLED
                # if self.tb_writer:
                #     self.tb_writer.add_scalar("train/loss", avg_loss, current_step)
                #     if accuracy is not None:
                #         self.tb_writer.add_scalar("train/accuracy", accuracy, current_step)
                #     self.tb_writer.add_scalar("train/tokens_seen", self.num_input_tokens_seen, current_step)

            # Eval
            if self.eval_inputs is not None and current_step % settings.eval_steps == 0:
                eval_metrics = self.evaluate_full_dataset()
                self._log_eval(current_step, eval_metrics)
                eval_loss = float(eval_metrics.loss)
                eval_acc_str = ""
                if eval_metrics.aux:
                    eval_token_acc = eval_metrics.aux.get("token_accuracy", 0)
                    eval_seq_acc = eval_metrics.aux.get("sequence_accuracy", 0)
                    eval_acc_str = f" | tok_acc={eval_token_acc:.4f} | seq_acc={eval_seq_acc:.4f}"
                    final_eval_accuracy = eval_token_acc
                self._log(f"  Eval @ {current_step}: loss={eval_loss:.4f}{eval_acc_str}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._update_best(current_step)

                # Early stopping based on no improvement
                if eval_metrics.aux and "token_accuracy" in eval_metrics.aux:
                    eval_acc = eval_metrics.aux["token_accuracy"]
                    if eval_acc > self.best_eval_accuracy:
                        self.best_eval_accuracy = eval_acc
                        self.no_improvement_counter = 0
                    else:
                        self.no_improvement_counter += 1

                    if self.no_improvement_counter >= self.early_stopping_patience:
                        self._log(f"Early stopping at step {current_step} due to no improvement in eval accuracy for {self.early_stopping_patience} steps.")
                        break

                # TensorBoard logging - DISABLED
                # if self.tb_writer:
                #     self.tb_writer.add_scalar("eval/loss", eval_loss, current_step)
                #     if eval_metrics.accuracy is not None:
                #         self.tb_writer.add_scalar("eval/accuracy", float(eval_metrics.accuracy), current_step)

            # Save checkpoints at every eval
            if self.eval_inputs is not None and current_step % settings.eval_steps == 0:
                self._save_checkpoint(current_step)

        # TensorBoard close - DISABLED
        # if self.tb_writer:
        #     self.tb_writer.close()

        def format_tokens(count):
            if count >= 1_000_000:
                return f"{count / 1_000_000:.2f}mn"
            elif count >= 1_000:
                return f"{count / 1_000:.0f}k"
            else:
                return str(count)

        tokens_in_fmt = format_tokens(self.num_input_tokens_seen)
        tokens_out_fmt = format_tokens(self.num_output_tokens_produced)
        n_unique_in = len(self.unique_input_tokens)

        self._log(
            f"Training complete. {num_steps} steps | "
            f"Tokens In: {tokens_in_fmt} ({n_unique_in} unique) | "
            f"Tokens Out: {tokens_out_fmt} | Final Eval Acc: {final_eval_accuracy:.4f}"
        )
        return history

    def save(self, path: str):
        """Serialize trainer state."""
        eqx.tree_serialise_leaves(path, self.state)

    def load(self, path: str):
        """Deserialize trainer state."""
        self.state = eqx.tree_deserialise_leaves(path, self.state)

    def evaluate(self, batch: Any) -> StepMetrics:
        """Run a JIT-compiled evaluation pass."""
        @eqx.filter_jit
        def _eval_pass(model, batch, key):
            loss, aux = loss_fn(model, batch, key)
            return StepMetrics(loss=loss, aux=aux)
        return _eval_pass(self.state.model, batch, self.state.key)

    def evaluate_full_dataset(self) -> StepMetrics:
        """Runs evaluation on the fixed eval dataset using instance variables."""
        if self.eval_inputs is None or self.eval_labels is None or self.eval_dataset_size is None:
            raise ValueError("Evaluation dataset not provided to Trainer.")

        eval_batch_size = 128
        num_batches = int(np.ceil(self.eval_dataset_size / eval_batch_size))

        total_loss = 0.0
        total_token_acc = 0.0
        total_seq_acc = 0.0

        task_metadata = self.task_metadata

        @jax.jit
        def eval_batch(m, inp, tgt, mask):
            return loss_fn(m, (inp, tgt, mask), jax.random.PRNGKey(0), task_metadata=task_metadata)

        for i in range(num_batches):
            start = i * eval_batch_size
            end = min(start + eval_batch_size, self.eval_dataset_size)
            if start >= end: break

            b_in = self.eval_inputs[start:end]
            b_tgt = self.eval_labels[start:end]
            b_mask = self.eval_loss_mask[start:end]

            l, metrics = eval_batch(self.state.model, b_in, b_tgt, b_mask)
            total_loss += float(l) * (end - start)
            total_token_acc += float(metrics["token_accuracy"]) * (end - start)
            total_seq_acc += float(metrics["sequence_accuracy"]) * (end - start)

        avg_loss = total_loss / self.eval_dataset_size
        avg_token_acc = total_token_acc / self.eval_dataset_size
        avg_seq_acc = total_seq_acc / self.eval_dataset_size

        return StepMetrics(
            loss=jnp.array(avg_loss),
            accuracy=jnp.array(avg_token_acc),
            aux={"token_accuracy": avg_token_acc, "sequence_accuracy": avg_seq_acc},
        )


# Loss Function
def loss_fn(model, batch, key, task_metadata=None):
    """Autoregressive loss with loss_mask as single source of truth.

    Args:
        model: The model to evaluate
        batch: Tuple of (inputs, targets, loss_mask)
        key: JAX random key
        task_metadata: Dict with optional 'output_vocab' key — a (vocab_size,) bool mask
                       of valid output tokens. If provided, logits for invalid tokens are
                       masked to -1e9 before softmax, sharpening the gradient signal.

    Returns:
        Tuple of (loss, metrics_dict)
    """
    inputs_np, targets_np, loss_mask_np = batch
    inputs = jnp.array(inputs_np)
    targets = jnp.array(targets_np)
    loss_mask = jnp.array(loss_mask_np)

    # Output vocab mask: (vocab_size,) bool, or None
    # Always include YIELD in the mask since loss includes the YIELD-predicting position.
    output_vocab_mask = None
    if task_metadata is not None and task_metadata.get("output_vocab") is not None:
        mask_np = task_metadata["output_vocab"].copy()
        mask_np[YIELD_TOKEN] = True
        output_vocab_mask = jnp.array(mask_np)  # (vocab_size,)

    def single_example(inp, tgt, mask):
        state = model.init_state()
        outputs, _ = model(inp, state)  # (T, vocab_size)

        # Apply output vocab mask: set invalid token logits to -1e9
        if output_vocab_mask is not None:
            outputs = jnp.where(output_vocab_mask[None, :], outputs, -1e9)

        log_probs = jax.nn.log_softmax(outputs, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, tgt[:, None], axis=-1).squeeze(-1)

        n_output_tokens = jnp.sum(mask) + 1e-5

        loss = -jnp.sum(target_log_probs * mask) / n_output_tokens

        correct = (jnp.argmax(outputs, axis=-1) == tgt)
        token_acc = jnp.sum(correct * mask) / n_output_tokens
        seq_acc = (jnp.sum(correct * mask) == jnp.sum(mask)).astype(jnp.float32)

        return loss, token_acc, seq_acc

    losses, token_accs, seq_accs = jax.vmap(single_example)(inputs, targets, loss_mask)

    return jnp.mean(losses), {
        "token_accuracy": jnp.mean(token_accs),
        "sequence_accuracy": jnp.mean(seq_accs),
    }

# Eval Function
def eval_fn(model, eval_dataset_size, eval_inputs, eval_labels, eval_loss_mask=None, task_metadata=None):
    """Runs evaluation on the fixed eval dataset."""
    eval_batch_size = 128
    num_batches = int(np.ceil(eval_dataset_size / eval_batch_size))

    total_loss = 0.0
    total_token_acc = 0.0
    total_seq_acc = 0.0

    @jax.jit
    def eval_batch(m, inp, tgt, mask):
        return loss_fn(m, (inp, tgt, mask), jax.random.PRNGKey(0))

    for i in range(num_batches):
        start = i * eval_batch_size
        end = min(start + eval_batch_size, eval_dataset_size)
        if start >= end: break

        b_in = eval_inputs[start:end]
        b_tgt = eval_labels[start:end]
        b_mask = eval_loss_mask[start:end]

        l, metrics = eval_batch(m, b_in, b_tgt, b_mask)
        total_loss += float(l) * (end - start)
        total_token_acc += float(metrics["token_accuracy"]) * (end - start)
        total_seq_acc += float(metrics["sequence_accuracy"]) * (end - start)

    avg_loss = total_loss / eval_dataset_size
    avg_token_acc = total_token_acc / eval_dataset_size
    avg_seq_acc = total_seq_acc / eval_dataset_size

    return StepMetrics(loss=jnp.array(avg_loss), accuracy=jnp.array(avg_token_acc),
                       aux={"token_accuracy": avg_token_acc, "sequence_accuracy": avg_seq_acc})
