import sys
import random
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

    # Task/Generation/Dataset configs
    task: str = Field(
        default="reverse_string",
        description=("Run `uv run inmaton-tasks` to see the list of available tasks.")
    )
    tasks: Optional[str] = Field(
        default=None,
        description="Comma-separated list of tasks for multi-task training (e.g. 'associative_recall,reverse_string'). Overrides --task when set."
    )
    max_seqlen: int = Field(default=40, description="Max sequence length for training")
    eval_max_seqlen: int = Field(default=100, description="Max sequence length for evaluation")
    task_kwargs: Optional[dict] = Field(default=None, description="Task-specific kwargs")
    eval_task_kwargs: Optional[dict] = Field(default=None, description="Task-specific kwargs")
    dataset_size: int = Field(default=10_000, description="Size of the training dataset.")
    eval_dataset_size: int = Field(default=1_000, description="Size of the evaluation dataset.")

    # Curriculum learning settings
    curriculum_type: Optional[str] = Field(
        default=None,
        description="Curriculum strategy: 'fixed', 'linear', 'adaptive', 'multitask', 'uniform' (default: None = no curriculum). 'uniform' matches Delétang et al. 2023: random length each step, on-the-fly batch generation."
    )
    curriculum_kwargs: Optional[dict] = Field(
        default=None,
        description="Curriculum hyperparameters (e.g., {'advance_threshold': 0.9, 'ema_decay': 0.95})"
    )


    # Model config
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
        description = "Additional model kwargs. Run `$ uv run inmaton-models` to see the required kwargs for each model."
    )
    embedding_dim: int = Field(default=16, description="Embedding dimension for the model. This becomes the common embedding dimension for all models.")
    embedding_type: str = Field(default='binary', description="Embedding type. Choices: 'binary', 'cosine', 'learnable', 'one_hot'. If 'one_hot', ignores embedding_dim.")
    hard_array_limit: int = Field(default=1024, description="Max Token Array size. This helps with autoregressive generation.")

    # Exhaustive Trainer args
    save_folder: str = Field(default="assets/training_runs", description="Root folder for saving logs and checkpoints.")
    run_name: Optional[str] = Field(default=None, description="Experiment name. Outputs are saved in {save_folder}/{run_name}")
    timeout: Optional[int]  = Field(default=None, description="Timeout in seconds for each run.")
    seed: Optional[int] = Field(default=None, description="Random seed for weight initialization and data shuffling.")
    eval_steps: int = Field(default=100, description="Evaluation frequency (run eval every N steps).")
    early_stopping_patience: int = Field(default=20, description="Number of eval rounds with no improvement before stopping.")
    save_limit: int = Field(default=5, description="Max checkpoints to retain excluing best model.")
    max_steps: int = Field(default=10_000, description="Total training iterations.")
    learning_rate: float = Field(default=3e-4, description="Initial learning rate.")
    optimizer: str = Field(default="adamw", description="Optimizer choice: 'adam', 'adamw', or 'sgd'.")
    optimizer_kwargs: Optional[dict] = Field(default=None, description="Optional optimizer arguments (e.g. {'weight_decay': 0.01})")
    batch_size: Optional[int] = Field(default=64, description="Batch size.")
    eval_batch_size: Optional[int] = Field(default=-1, description="By default (-1) send all items in one go.")
    precision: str = Field(default="mixed-bf16-fp32", description="Compute precision: 'fp32', 'bf16', or 'mixed-bf16-fp32'.")

    @model_validator(mode='after')
    def set_correct_default(self) -> "Settings":
        if self.run_name is None:
            self.run_name = randomname.get_name()
        if self.seed is None:
            self.seed = random.randint(0, 1000)
        print(f"Run name: {ANSI.bold(self.run_name)}")
        print(f"Using seed: {ANSI.bold(self.seed)}")
        print(f"Hard array limit: {ANSI.bold(str(self.hard_array_limit))}")
        return self

def get_logger(name: str = "industrial_automaton", log_level: str = "INFO") -> logging.Logger:
    """Returns a logger object"""
    lvl = log_level.upper()
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
