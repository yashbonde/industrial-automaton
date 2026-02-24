import sys
import logging
import randomname
from typing import Optional
from pydantic import model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from industrial_automaton.utils import ANSI

class Settings(BaseSettings):
    """CLI for training neural automata models. Part of Industrial Automaton project by @yashbonde.

    Read more at yashbonde.com/blogs/automata/0-prologue-start

    This CLI uses JAX as the underlying system."""

    model_config = SettingsConfigDict(
        env_prefix="INMATON_",
        case_sensitive=False,
        env_file=".env",
        extra="allow",
        cli_parse_args=True,
    )

    def __init__(self, **data):
        # Disable CLI parsing when in pytest to avoid conflicts
        is_pytest = "pytest" in sys.modules or any("pytest" in arg for arg in sys.argv)
        if is_pytest and "_cli_parse_args" not in data:
            data["_cli_parse_args"] = []
        super().__init__(**data)

    # General configuration
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    seed: int = Field(default=4, description="Random seed for reproducibility")

    # Task/Generation limits
    task: str = Field(
        default="reverse_string", 
        description=(
            "Task to perform. Available tasks categorized by complexity:\n"
            "  - REGULAR: parity_check, even_pairs, modular_arithmetic, cycle_navigation\n"
            "  - CONTEXT-FREE: dyck_n, reverse_string, stack_manipulation, solve_equation, nested_modular_arithmetic\n"
            "  - CONTEXT-SENSITIVE: binary_addition, binary_multiplication, square_root, repeat_copy_n, duplicate_string, count_n, associative_recall, odds_first\n"
            "  - ALGORITHMIC: sort, python_execution, graph_traversal, shortest_path, tsp, mst_prim"
        )
    )
    n: Optional[int] = Field(default=None, description="Task parameter 'n' (e.g. number of bracket types for dyck_n, number of symbol types for count_n)")
    tr_max_seqlen: int = Field(default=40, description="Max sequence length for training")
    tr_task_kwargs: Optional[dict] = Field(default=None, description="Task-specific kwargs")
    tr_eval_max_seqlen: int = Field(default=100, description="Max sequence length for evaluation")
    tr_eval_task_kwargs: Optional[dict] = Field(default=None, description="Task-specific kwargs")

    # Model
    model: str = Field(
        default="baby_ntm", 
        description=(
            "Model architecture. Choices:\n"
            "  - baby_ntm: LSTM + simple 5-op deterministic memory\n"
            "  - suzgun_stack_rnn: RNN + differentiable stack (best for Context-Free)\n"
            "  - tape_rnn: LSTM + differentiable moving-head tape (best for Context-Sensitive)\n"
            "  - transformer: Standard decoder-only causal Transformer\n"
            "  - lstm: Baseline gated RNN without external memory"
        )
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description = "Additional model kwargs. Run uv run inmaton-models to see the required kwargs for each model."
    )

    # Exhaustive Trainer args
    tr_run_name: Optional[str] = Field(default=None, description="Experiment name. Outputs saved in {save_folder}/{run_name}")
    tr_eval_steps: int = Field(default=100, description="Evaluation frequency (run eval every N steps).")
    tr_seed: Optional[int] = Field(default=None, description="Random seed for weight initialization and data shuffling.")
    tr_save_folder: str = Field(default="assets/training_runs", description="Root folder for saving logs and checkpoints.")
    tr_save_steps: int = Field(default=1000, description="Checkpoint save frequency.")
    tr_save_limit: int = Field(default=5, description="Max checkpoints to retain excluing best model.")
    tr_num_train_epochs: int = Field(default=1, description="Number of training epochs (1 is usually enough for synthetic tasks).")
    tr_max_steps: int = Field(default=10_000, description="Total training iterations.")
    tr_lr: float = Field(default=1e-3, description="Initial learning rate.")
    tr_lr_scheduler: Optional[str] = Field(default=None, description="LR schedule: 'constant', 'cosine', or 'linear'.")
    tr_lr_scheduler_kwargs: Optional[dict] = Field(default=None, description="Optional scheduler arguments (e.g. {'end_value': 1e-5})")
    tr_warmup_steps: int = Field(default=0, description="Linear warmup steps.")
    tr_optimizer: str = Field(default="adamw", description="Optimizer choice: 'adam', 'adamw', or 'sgd'.")
    tr_optimizer_kwargs: Optional[dict] = Field(default=None, description="Optional optimizer arguments (e.g. {'weight_decay': 0.01})")
    tr_gradient_accumulation_steps: int = Field(default=1, description="Steps to accumulate before an update (effectively batch_size multiplier).")
    tr_max_grad_norm: float = Field(default=1.0, description="Threshold for gradient clipping.")
    tr_precision: str = Field(default="mixed-bf16-fp32", description="Compute precision: 'fp32', 'bf16', or 'mixed-bf16-fp32'.")
    tr_logging_steps: int = Field(default=10, description="Logging frequency.")
    tr_tensorboard: bool = Field(default=False, description="Enable TensorBoard logging.")
    tr_tensorboard_log_dir: Optional[str] = Field(default=None, description="Required when logging using tensorboard. Path to the TensorBoard log directory.")

    @model_validator(mode='after')
    def set_correct_default(self) -> "Settings":
        if self.tr_run_name is None:
            self.tr_run_name = randomname.get_name()
        if self.seed is None:
            self.seed = random.randint(0, 1000)
        return self

    @model_validator(mode='after')
    def validate_task_and_seqlen(self) -> "Settings":
        # We handle task naming flexibility
        if not self.task.startswith("generate_"):
            task_key = f"generate_{self.task}"
        else:
            task_key = self.task

        _MAX_LENGTHS = {
            "generate_binary_addition": 64,
            "generate_binary_multiplication": 32,
            "generate_square_root": 19,
            "generate_repeat_copy_n": 1000,
            "generate_duplicate_string": 3333,
            "generate_deduplicate_inputs": 3333,
            "generate_missing_duplicate": 5000,
            "generate_odds_first": 5000,
            "generate_associative_recall": 18,
            "generate_n_back": 4999,
            "generate_python_execution": 999,
            "generate_sort": 5000,
            "generate_stack_manipulation": 5000,
            "generate_reverse_string": 5000,
            "generate_dyck_n": 5000,
            "generate_parity": 5000,
        }
        
        limit = _MAX_LENGTHS.get(task_key, 1000)
        if self.tr_max_seqlen > limit:
            print(f"Setting maxlen for {ANSI.bold(task_key)} to {ANSI.bold(limit)} (was {self.tr_max_seqlen}).")
            self.tr_max_seqlen = limit
        return self

# Create global settings instance
settings = Settings()

def get_logger(name: str = "industrial_automaton") -> logging.Logger:
    """Returns a logger object"""
    lvl = settings.log_level.upper()
    logger = logging.getLogger(name)
    # Clear handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(getattr(logging, lvl))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(log_handler)
    return logger

logger = get_logger()
