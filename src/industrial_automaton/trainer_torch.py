"""PyTorch trainer — mirrors trainer_jx.py but uses torch + MPS.

Reproducibility strategy:
  - torch.manual_seed(seed) sets global RNG (CPU + MPS)
  - A torch.Generator(device='cpu') seeded from settings.seed is passed
    to model constructors so weight initialization is deterministic and
    independent of any other global RNG usage.
  - torch.use_deterministic_algorithms(True, warn_only=True) catches
    non-deterministic ops when possible.

MPS notes:
  - Models and tensors are moved to `device` (mps / cuda / cpu) via .to()
  - torch.roll, torch.where, nn.LSTMCell etc. are all MPS-supported
  - Gradient clipping uses torch.nn.utils.clip_grad_norm_
  - Checkpoints saved via torch.save / torch.load
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from industrial_automaton.config import Settings
from industrial_automaton.vocab import PAD, YIELD as YIELD_TOKEN


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_generator(seed: int, device: torch.device = torch.device("cpu")) -> torch.Generator:
    """Create a seeded CPU generator (weight init must happen on CPU)."""
    g = torch.Generator(device=torch.device("cpu"))
    g.manual_seed(seed)
    return g


# ── Divergence monitor ────────────────────────────────────────────────────────

class TrainingDivergedError(Exception):
    pass


class DivergenceMonitor:
    """Same signals as the JAX DivergenceMonitor."""

    def __init__(
        self,
        grad_explosion_threshold: float = 1000.0,
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

        self.vanishing_streak = 0
        self.loss_ema = None
        self.plateau_count = 0

    def check(self, loss: float, grad_norm: float, step: int):
        if np.isnan(loss) or np.isinf(loss):
            raise TrainingDivergedError(f"Loss NaN/Inf at step {step}: {loss}")

        if grad_norm > self.grad_explosion_threshold:
            raise TrainingDivergedError(f"Gradient explosion: norm={grad_norm:.2f} at step {step}")

        if grad_norm < 1e-6:
            self.vanishing_streak += 1
            if self.vanishing_streak > self.vanishing_patience:
                raise TrainingDivergedError(
                    f"Gradient vanishing for {self.vanishing_streak} steps at step {step}"
                )
        else:
            self.vanishing_streak = 0

        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            old_ema = self.loss_ema
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * loss
            if abs(self.loss_ema - old_ema) < self.plateau_threshold and self.loss_ema > 1e-4:
                self.plateau_count += 1
                if self.plateau_count > self.plateau_patience:
                    raise TrainingDivergedError(
                        f"Loss plateaued at {self.loss_ema:.4f} for {self.plateau_count} steps at step {step}"
                    )
            else:
                self.plateau_count = 0


# ── Step metrics ──────────────────────────────────────────────────────────────

class StepMetrics(NamedTuple):
    loss: float
    accuracy: Optional[float] = None
    aux: Optional[Dict[str, float]] = None


# ── Optimizer builder ─────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, settings: Settings) -> torch.optim.Optimizer:
    opt_name = settings.optimizer.lower()
    lr = settings.learning_rate
    kwargs = settings.optimizer_kwargs or {}
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}. Choose from adam, adamw, sgd")


# ── Loss function ─────────────────────────────────────────────────────────────

def loss_fn(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_vocab_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Autoregressive cross-entropy loss with PAD/output masking.

    Vectorized version: processes the entire batch in one pass.
    """
    inputs, targets, loss_mask = batch
    B, T = inputs.shape

    state = model.init_state(batch_size=B, device=inputs.device)
    logits, _ = model(inputs, state)  # (B, T, vocab_size)

    if output_vocab_mask is not None:
        logits = logits.masked_fill(~output_vocab_mask.view(1, 1, -1), -1e9)

    # Flatten for cross-entropy
    logits_flat = logits.view(-1, logits.shape[-1])
    targets_flat = targets.view(-1)
    mask_flat = loss_mask.view(-1).float()

    log_probs = F.log_softmax(logits_flat, dim=-1)
    target_lp = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

    loss_per_token = -target_lp * mask_flat
    
    # Per-example sequence accuracy
    preds_flat = logits_flat.argmax(dim=-1)
    correct_flat = (preds_flat == targets_flat).float()
    
    # Reshape back to (B, T) to compute sequence accuracy
    correct = (correct_flat * mask_flat).view(B, T)
    mask = mask_flat.view(B, T)
    
    # Seq acc: all masked tokens in an example must be correct
    # If mask is all 0 (e.g. padding only), we'll handle that
    seq_correct = (correct.sum(dim=1) == mask.sum(dim=1)).float()
    
    # Token accuracy
    num_output_tokens = mask_flat.sum() + 1e-5
    token_acc = (correct_flat * mask_flat).sum() / num_output_tokens
    
    # Overall loss
    loss = loss_per_token.sum() / num_output_tokens
    
    return loss, {
        "token_accuracy": token_acc.item(),
        "sequence_accuracy": seq_correct.mean().item()
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class TorchTrainer:
    """Feature-rich PyTorch trainer driven by Settings config.

    Mirrors the JAX Trainer API:
      - fit(data_generator) training loop
      - evaluate_full_dataset() for eval
      - save / load checkpoints
      - Divergence monitoring, early stopping, curriculum support
      - Logging to file + stdout + eval_logs.jsonl
    """

    def __init__(
        self,
        model: nn.Module,
        settings: Settings,
        task_metadata: Optional[Dict[str, Any]] = None,
        curriculum: Optional[Any] = None,
        enable_divergence_monitor: bool = True,
        eval_inputs: Optional[np.ndarray] = None,
        eval_labels: Optional[np.ndarray] = None,
        eval_loss_mask: Optional[np.ndarray] = None,
        eval_dataset_size: Optional[int] = None,
    ):
        self.settings        = settings
        self.task_metadata   = task_metadata
        self.curriculum      = curriculum
        self.curriculum_bound = 1.0  # used if curriculum_fn provided to fit()

        # Reproducibility
        seed = settings.seed if settings.seed is not None else 0
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Device
        self.device = get_device()

        # Move model to device
        self.model = model.to(self.device)

        # Precision
        if settings.precision in ("bf16", "mixed-bf16-fp32"):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = None  # fp32

        # Output vocab mask
        self.output_vocab_mask = None
        if task_metadata is not None and task_metadata.get("output_vocab") is not None:
            mask_np = task_metadata["output_vocab"].copy()
            mask_np[YIELD_TOKEN] = True
            self.output_vocab_mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)

        # Eval data
        self.eval_inputs       = eval_inputs
        self.eval_labels       = eval_labels
        self.eval_loss_mask    = eval_loss_mask
        self.eval_dataset_size = eval_dataset_size

        # Optimizer
        self.optimizer = build_optimizer(self.model, settings)

        # Monitoring
        self.divergence_monitor      = DivergenceMonitor() if enable_divergence_monitor else None
        self.best_eval_accuracy      = -float("inf")
        self.no_improvement_counter  = 0
        self.early_stopping_patience = getattr(settings, "early_stopping_patience", 20)

        # Token counting
        self.num_input_tokens_seen      = 0
        self.num_output_tokens_produced = 0
        self.unique_input_tokens: set   = set()

        # Paths
        from pathlib import Path
        self.run_dir       = Path(settings.save_folder) / settings.run_name
        self.ckpt_dir      = self.run_dir / "ckpt"
        self.log_file      = self.run_dir / "logs"
        self.eval_log_file = self.run_dir / "eval_logs.jsonl"
        self.config_file   = self.run_dir / "config.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.config_file.write_text(settings.model_dump_json(indent=2))

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def _log_eval(self, step: int, metrics: StepMetrics):
        record = {"step": step, "loss": metrics.loss, "timestamp": time.time()}
        if metrics.accuracy is not None:
            record["accuracy"] = metrics.accuracy
        with open(self.eval_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, step: int):
        step_dir = self.ckpt_dir / f"step-{step}"
        step_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step":                 step,
            },
            step_dir / "state.pt",
        )
        # Rotate: keep only save_limit checkpoints
        best_link   = self.ckpt_dir / "best"
        best_target = best_link.resolve() if best_link.is_symlink() else None

        step_dirs = sorted(
            [d for d in self.ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if len(step_dirs) > self.settings.save_limit:
            import shutil
            for d in step_dirs[: len(step_dirs) - self.settings.save_limit]:
                if best_target and d.resolve() == best_target:
                    continue
                shutil.rmtree(d)

    def _update_best(self, step: int):
        best_link = self.ckpt_dir / "best"
        target    = self.ckpt_dir / f"step-{step}"
        if best_link.is_symlink() or best_link.exists():
            best_link.unlink()
        best_link.symlink_to(target.resolve())

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ── Training ──────────────────────────────────────────────────────────────

    def _to_device(self, batch):
        """Move a (inputs, targets, loss_mask) tuple to self.device as long tensors."""
        inputs, targets, loss_mask = batch
        return (
            torch.as_tensor(inputs,    dtype=torch.long, device=self.device),
            torch.as_tensor(targets,   dtype=torch.long, device=self.device),
            torch.as_tensor(loss_mask, dtype=torch.bool, device=self.device),
        )

    def _train_step(self, batch) -> Tuple[float, Dict[str, float]]:
        """Single training step. Returns (loss_val, metrics_dict)."""
        self.model.train()
        self.optimizer.zero_grad()

        device_batch = self._to_device(batch)

        if self.amp_dtype is not None:
            # MPS autocast: use bfloat16 for forward, fp32 for param updates
            autocast_device = "mps" if self.device.type == "mps" else self.device.type
            with torch.autocast(device_type=autocast_device, dtype=self.amp_dtype):
                loss, metrics = loss_fn(self.model, device_batch, self.output_vocab_mask)
        else:
            loss, metrics = loss_fn(self.model, device_batch, self.output_vocab_mask)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
        self.optimizer.step()

        return loss.item(), metrics, grad_norm

    def fit(
        self,
        data_generator: Union[Iterator, Callable[[float], Any]],
        curriculum_fn: Optional[Callable[[float, StepMetrics], float]] = None,
    ) -> List[StepMetrics]:
        """Training loop — mirrors trainer_jx.py Trainer.fit()."""
        settings   = self.settings
        num_steps  = settings.max_steps
        history    = []
        start_time = time.time()
        is_iter    = hasattr(data_generator, "__next__")
        best_eval_loss = float("inf")
        final_eval_accuracy = 0.0

        self._log(f"Starting training for {num_steps} steps (device={self.device})")

        for step_idx in range(num_steps):
            # Get batch
            if is_iter:
                batch = next(data_generator)
            else:
                batch = data_generator(self.curriculum_bound)

            # Token accounting
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                inp_b, _, mask_b = batch
                if hasattr(inp_b, "shape") and len(np.shape(inp_b)) >= 2:
                    inp_arr = np.asarray(inp_b)
                    self.num_input_tokens_seen += int(inp_arr.size)
                    self.unique_input_tokens.update(int(x) for x in np.unique(inp_arr))
                if hasattr(mask_b, "__len__"):
                    self.num_output_tokens_produced += int(np.sum(mask_b))

            loss_val, metrics, grad_norm = self._train_step(batch)

            if self.divergence_monitor:
                self.divergence_monitor.check(loss_val, grad_norm, step_idx)

            current_step = step_idx + 1
            tok_acc  = metrics.get("token_accuracy")
            seq_acc  = metrics.get("sequence_accuracy")
            step_metrics = StepMetrics(loss=loss_val, accuracy=tok_acc, aux=metrics)
            history.append(step_metrics)

            # Timeout
            if settings.timeout is not None and (time.time() - start_time) > settings.timeout:
                self._log(f"Timeout ({settings.timeout}s) reached at step {current_step}.")
                break

            # Curriculum
            if curriculum_fn is not None:
                new_bound = curriculum_fn(self.curriculum_bound, step_metrics)
                self.curriculum_bound = new_bound

            # Logging every 10 steps
            if current_step % 10 == 0:
                elapsed = time.time() - start_time
                acc_str = ""
                if tok_acc is not None:
                    acc_str = f" | TokAcc: {tok_acc:.4f}"
                if seq_acc is not None:
                    acc_str += f" | SeqAcc: {seq_acc:.4f}"

                def _fmt(n):
                    if n >= 1_000_000: return f"{n/1_000_000:.2f}mn"
                    if n >= 1_000:     return f"{n/1_000:.0f}k"
                    return str(n)

                self._log(
                    f"Step {current_step:05d} | Loss: {loss_val:.4f}{acc_str}"
                    f" | Curr: {self.curriculum_bound:.2f}"
                    f" | Tokens In: {_fmt(self.num_input_tokens_seen)}"
                    f" | Tokens Out: {_fmt(self.num_output_tokens_produced)}"
                    f" | Time: {elapsed:.2f}s"
                )

            # Eval + checkpoint
            if self.eval_inputs is not None and current_step % settings.eval_steps == 0:
                eval_metrics = self.evaluate_full_dataset()
                self._log_eval(current_step, eval_metrics)
                eval_loss = eval_metrics.loss
                eval_acc_str = ""
                if eval_metrics.aux:
                    etok = eval_metrics.aux.get("token_accuracy", 0)
                    eseq = eval_metrics.aux.get("sequence_accuracy", 0)
                    eval_acc_str = f" | tok_acc={etok:.4f} | seq_acc={eseq:.4f}"
                    final_eval_accuracy = eseq
                self._log(f"  Eval @ {current_step}: loss={eval_loss:.4f}{eval_acc_str}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._update_best(current_step)

                if eval_metrics.aux and "token_accuracy" in eval_metrics.aux:
                    ea = eval_metrics.aux["token_accuracy"]
                    if ea > self.best_eval_accuracy:
                        self.best_eval_accuracy = ea
                        self.no_improvement_counter = 0
                    else:
                        self.no_improvement_counter += 1

                    if self.no_improvement_counter >= self.early_stopping_patience:
                        self._log(
                            f"Early stopping at step {current_step} — no improvement for "
                            f"{self.early_stopping_patience} evals."
                        )
                        break

                self._save_checkpoint(current_step)

        def _fmt(n):
            if n >= 1_000_000: return f"{n/1_000_000:.2f}mn"
            if n >= 1_000:     return f"{n/1_000:.0f}k"
            return str(n)

        self._log(
            f"Training complete. {num_steps} steps | "
            f"Tokens In: {_fmt(self.num_input_tokens_seen)} "
            f"({len(self.unique_input_tokens)} unique) | "
            f"Tokens Out: {_fmt(self.num_output_tokens_produced)} | "
            f"Final Eval Acc: {final_eval_accuracy:.4f}"
        )
        return history

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_full_dataset(self) -> StepMetrics:
        if self.eval_inputs is None:
            raise ValueError("No eval dataset provided.")

        self.model.eval()
        eval_batch_size = 128
        n = self.eval_dataset_size
        num_batches = int(np.ceil(n / eval_batch_size))

        total_loss, total_tok, total_seq = 0.0, 0.0, 0.0

        with torch.no_grad():
            for i in range(num_batches):
                s = i * eval_batch_size
                e = min(s + eval_batch_size, n)
                if s >= e:
                    break

                b_in   = self.eval_inputs[s:e]
                b_tgt  = self.eval_labels[s:e]
                b_mask = self.eval_loss_mask[s:e]

                device_batch = self._to_device((b_in, b_tgt, b_mask))
                loss, metrics = loss_fn(self.model, device_batch, self.output_vocab_mask)

                cnt = e - s
                total_loss += loss.item() * cnt
                total_tok  += metrics["token_accuracy"]  * cnt
                total_seq  += metrics["sequence_accuracy"] * cnt

        avg_loss = total_loss / n
        avg_tok  = total_tok  / n
        avg_seq  = total_seq  / n

        return StepMetrics(
            loss=avg_loss,
            accuracy=avg_tok,
            aux={"token_accuracy": avg_tok, "sequence_accuracy": avg_seq},
        )
