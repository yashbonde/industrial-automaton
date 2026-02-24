import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from industrial_automaton.config import Settings


class TrainState(eqx.Module):
    """Unified state for the trainer, fully compatible with JAX transformations."""
    model: eqx.Module
    opt_state: optax.OptState
    key: jax.Array
    step: jnp.ndarray
    curriculum_bound: jnp.ndarray


class StepMetrics(NamedTuple):
    """Container for metrics from a single training step."""
    loss: jnp.ndarray
    accuracy: Optional[jnp.ndarray] = None
    aux: Optional[Dict[str, jnp.ndarray]] = None


def build_optimizer(settings: Settings) -> optax.GradientTransformation:
    """Build an optax optimizer chain from settings."""
    # LR schedule
    lr = settings.tr_lr
    sched_name = (settings.tr_lr_scheduler or "constant").lower()

    if sched_name == "constant":
        schedule = lr
    elif sched_name == "cosine":
        kwargs = dict(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=int(settings.tr_warmup_steps),
            decay_steps=int(settings.tr_max_steps),
            end_value=0.0,
        )
        if settings.tr_lr_scheduler_kwargs:
            kwargs.update(settings.tr_lr_scheduler_kwargs)
        schedule = optax.warmup_cosine_decay_schedule(**kwargs)
    elif sched_name == "linear":
        schedule = optax.linear_schedule(
            init_value=lr, end_value=0.0, transition_steps=settings.tr_max_steps
        )
    else:
        raise ValueError(f"Unknown lr scheduler: {sched_name}")

    # Optimizer
    opt_name = settings.tr_optimizer.lower()
    opt_map = {"adam": optax.adam, "adamw": optax.adamw, "sgd": optax.sgd}
    if opt_name not in opt_map:
        raise ValueError(f"Unknown optimizer: {opt_name}. Choose from {list(opt_map)}")
    opt_kwargs = settings.tr_optimizer_kwargs or {}
    optimizer = opt_map[opt_name](learning_rate=schedule, **opt_kwargs)

    return optax.chain(optax.clip_by_global_norm(settings.tr_max_grad_norm), optimizer)


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


class JAXTrainer:
    """Feature-rich JAX trainer driven by Settings config.

    Features: checkpoint saving with best-symlink and rotation, TensorBoard,
    file+CLI logging, eval JSONL logging, LR scheduler, gradient accumulation,
    gradient clipping, and mixed precision.
    """

    def __init__(
        self,
        model: eqx.Module,
        loss_fn: Callable[[eqx.Module, Any, jax.Array], Tuple[jnp.ndarray, Dict[str, Any]]],
        settings: Settings,
    ):
        self.settings = settings
        self.loss_fn = loss_fn
        self.num_input_tokens_seen = 0

        # Build optimizer
        self.optimizer = build_optimizer(settings)

        # Init state
        key = jax.random.PRNGKey(settings.tr_seed)
        _, state_key = jax.random.split(key)

        # Handle pure bf16 training
        if settings.tr_precision == "bf16":
            model = _cast_bf16(model)

        self.state = TrainState(
            model=model,
            opt_state=self.optimizer.init(eqx.filter(model, eqx.is_array)),
            key=state_key,
            step=jnp.array(0, dtype=jnp.int32),
            curriculum_bound=jnp.array(1.0, dtype=jnp.float32),
        )

        # Run directory
        self.run_dir = Path(settings.tr_save_folder) / settings.tr_run_name
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

        # TensorBoard
        self.tb_writer = None
        if settings.tr_tensorboard:
            try:
                from tensorboardX import SummaryWriter
                self.tb_writer = SummaryWriter(str(self.tb_dir))
            except ImportError:
                self._log("WARNING: tensorboardX not installed, disabling TensorBoard")

            if settings.tr_tensorboard_log_dir:
                tb_link = Path(settings.tr_tensorboard_log_dir)
                tb_link.parent.mkdir(parents=True, exist_ok=True)
                if tb_link.is_symlink() or tb_link.exists():
                    tb_link.unlink()
                tb_link.symlink_to(self.tb_dir.resolve())

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
        precision = self.settings.tr_precision

        if precision == "mixed-bf16-fp32":
            model_compute = _cast_bf16(model)
        else:
            # For "bf16" or "fp32", model is already in correct precision from __init__
            model_compute = model

        @eqx.filter_value_and_grad(has_aux=True)
        def compute(m, batch, key):
            loss, aux = self.loss_fn(m, batch, key)
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
            curriculum_bound=state.curriculum_bound,
        )

    def _save_checkpoint(self, step: int):
        """Save checkpoint and rotate old ones."""
        step_dir = self.ckpt_dir / f"step-{step}"
        step_dir.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(str(step_dir / "state.eqx"), self.state)

        # Rotate: find all step dirs, keep only tr_save_limit most recent
        best_link = self.ckpt_dir / "best"
        best_target = None
        if best_link.is_symlink():
            best_target = best_link.resolve()

        step_dirs = sorted(
            [d for d in self.ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
            key=lambda d: int(d.name.split("-")[1]),
        )

        if len(step_dirs) > self.settings.tr_save_limit:
            to_remove = step_dirs[: len(step_dirs) - self.settings.tr_save_limit]
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
        eval_fn: Optional[Callable[[eqx.Module], StepMetrics]] = None,
        curriculum_fn: Optional[Callable[[float, StepMetrics], float]] = None,
    ) -> List[StepMetrics]:
        """Training loop.

        Args:
            data_generator: Iterator yielding batches, or callable(curriculum_bound) -> batch.
            eval_fn: Optional callable(model) -> StepMetrics for evaluation.
            curriculum_fn: Optional callable(current_bound, metrics) -> new_bound.
        """
        settings = self.settings
        num_steps = settings.tr_max_steps
        accum_steps = settings.tr_gradient_accumulation_steps
        history = []
        start_time = time.time()
        is_iterator = isinstance(data_generator, Iterator)
        best_eval_loss = float("inf")

        self._log(f"Starting training for {num_steps} steps (accum={accum_steps})")

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
                    batch = data_generator(float(self.state.curriculum_bound))

                # Token counting
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    b = batch[0]
                    if hasattr(b, "shape") and len(b.shape) >= 2:
                        self.num_input_tokens_seen += int(b.shape[0] * b.shape[1])

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

            # Apply
            self.state = self._jit_apply_grads(self.state, avg_grads)

            accuracy = avg_aux.get("accuracy")
            acc_val = jnp.array(accuracy) if accuracy is not None else None
            metrics = StepMetrics(loss=jnp.array(avg_loss), accuracy=acc_val, aux=avg_aux or None)
            history.append(metrics)

            current_step = int(self.state.step)

            # Curriculum
            if curriculum_fn:
                new_bound = curriculum_fn(float(self.state.curriculum_bound), metrics)
                if new_bound != float(self.state.curriculum_bound):
                    self.update_curriculum(new_bound)

            # Logging
            if current_step % settings.tr_logging_steps == 0:
                elapsed = time.time() - start_time
                acc_str = f" | Acc: {accuracy:.4f}" if accuracy is not None else ""
                curr_str = f" | Curr: {self.state.curriculum_bound:.2f}"
                
                # Calculate epoch
                steps_per_epoch = max(1, settings.tr_max_steps // max(1, settings.tr_num_train_epochs))
                epoch = (current_step - 1) // steps_per_epoch + 1
                
                msg = (
                    f"Epoch {epoch:02d} | Step {current_step:05d} | Loss: {avg_loss:.4f}{acc_str}{curr_str}"
                    f" | Tokens: {self.num_input_tokens_seen} | Time: {elapsed:.2f}s"
                )
                self._log(msg)

                if self.tb_writer:
                    self.tb_writer.add_scalar("train/loss", avg_loss, current_step)
                    if accuracy is not None:
                        self.tb_writer.add_scalar("train/accuracy", accuracy, current_step)
                    self.tb_writer.add_scalar("train/tokens_seen", self.num_input_tokens_seen, current_step)

            # Eval
            if eval_fn and current_step % settings.tr_eval_steps == 0:
                eval_metrics = eval_fn(self.state.model)
                self._log_eval(current_step, eval_metrics)
                eval_loss = float(eval_metrics.loss)
                self._log(f"  Eval @ {current_step}: loss={eval_loss:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._update_best(current_step)
                if self.tb_writer:
                    self.tb_writer.add_scalar("eval/loss", eval_loss, current_step)
                    if eval_metrics.accuracy is not None:
                        self.tb_writer.add_scalar("eval/accuracy", float(eval_metrics.accuracy), current_step)

            # Save
            if current_step % settings.tr_save_steps == 0:
                self._save_checkpoint(current_step)

        if self.tb_writer:
            self.tb_writer.close()

        self._log(f"Training complete. {num_steps} steps, {self.num_input_tokens_seen} tokens seen.")
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
            loss, aux = self.loss_fn(model, batch, key)
            return StepMetrics(loss=loss, aux=aux)

        return _eval_pass(self.state.model, batch, self.state.key)
