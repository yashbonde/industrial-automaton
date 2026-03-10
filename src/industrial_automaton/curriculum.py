"""Functional State Curriculum System for Industrial Automaton.

Treats training difficulty as a control system, where the curriculum controller
observes model performance and dynamically adjusts data difficulty to maximize
learning rate.

Philosophy: Training as a Control System
========================================
State-Space Model:
    Input: CurriculumState + Metrics (loss, accuracy)
    Transition: update() function (controller logic)
    Output: Next CurriculumState

The curriculum is a Controller that:
- Observes the Plant (neural network performance)
- Adjusts the Control Input (data difficulty)
- Maximizes the Objective (learning rate, convergence)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union

import equinox as eqx
import numpy as np


class CurriculumState(eqx.Module):
    """Functional state for curriculum tracking.

    All curriculum state is pure, immutable, and JIT-compatible.
    Use equinox.tree_at() for functional updates.
    """
    step: int
    current_bound: Union[int, float]  # Length, vocab_size, etc.
    min_bound: int  # Hard minimum
    max_bound: int  # Hard maximum
    task_progress: Optional[Dict[str, float]] = None  # For multi-task
    smoothed_accuracy: Optional[float] = None  # For adaptive
    consecutive_successes: int = 0  # For adaptive


class CurriculumStrategy(Protocol):
    """Interface all curriculum strategies must implement.

    All strategies are control-theoretic: they observe metrics and output
    new curriculum state to guide training difficulty.
    """

    def update(self, state: CurriculumState, metrics: Dict[str, float]) -> CurriculumState:
        """Transition function: state + metrics → new_state.

        Args:
            state: Current curriculum state
            metrics: {"loss": float, "accuracy": float, "task_id": Optional[str]}

        Returns:
            Updated curriculum state
        """
        ...

    def get_params(self, state: CurriculumState) -> Dict[str, Any]:
        """Extract data generator parameters from state.

        Returns:
            {"length": int, "task_id": str, ...}
        """
        ...


@dataclass
class FixedCurriculum:
    """No adaptation - always returns initial configuration.

    Use case: Baseline evaluations, ablation studies.

    Attributes:
        fixed_params: Dictionary of parameters to return (e.g., {"length": 50})
    """

    fixed_params: Dict[str, Any]

    def update(self, state: CurriculumState, metrics: Dict[str, float]) -> CurriculumState:
        """Increment step but don't change bound."""
        return CurriculumState(
            step=state.step + 1,
            current_bound=state.current_bound,
            min_bound=state.min_bound,
            max_bound=state.max_bound,
            task_progress=state.task_progress,
            smoothed_accuracy=state.smoothed_accuracy,
            consecutive_successes=state.consecutive_successes,
        )

    def get_params(self, state: CurriculumState) -> Dict[str, Any]:
        """Return fixed parameters."""
        return self.fixed_params


@dataclass
class LinearCurriculum:
    """Linear increase: bound = min(max, initial + (step // freq) * amount).

    Use case: "RegularIncrease" from Chomsky Hierarchy paper.

    Logic: Increase difficulty linearly with training steps.

    Example:
        initial_bound=5, max_bound=100, increase_freq=1000, increase_amount=5
        - Step 0-999: length=5
        - Step 1000-1999: length=10
        - Step 2000-2999: length=15
        - Step 19000+: length=100 (capped)

    Attributes:
        initial_bound: Starting difficulty level
        max_bound: Maximum difficulty level (cap)
        increase_freq: Steps between increases
        increase_amount: How much to increase per step
    """

    initial_bound: int
    max_bound: int
    increase_freq: int  # Steps between increases
    increase_amount: int  # How much to increase

    def update(self, state: CurriculumState, metrics: Dict[str, float]) -> CurriculumState:
        """Update bound based on linear schedule."""
        new_bound = min(
            self.max_bound,
            self.initial_bound + (state.step // self.increase_freq) * self.increase_amount
        )
        return CurriculumState(
            step=state.step + 1,
            current_bound=new_bound,
            min_bound=state.min_bound,
            max_bound=state.max_bound,
            task_progress=state.task_progress,
            smoothed_accuracy=state.smoothed_accuracy,
            consecutive_successes=state.consecutive_successes,
        )

    def get_params(self, state: CurriculumState) -> Dict[str, Any]:
        """Extract current length from state."""
        return {"length": int(state.current_bound)}


@dataclass
class AdaptiveCurriculum:
    """Performance-based adaptation with backoff.

    Use case: "BetterCurriculum" from Neural GPU paper.

    Logic:
        - Advance: If accuracy > threshold for N consecutive steps
        - Backoff: If loss > danger_threshold at max length
        - Smoothing: EMA to prevent noise-driven jumping

    Why backoff is critical:
        - Without it: Model diverges at max length, NaN loss, training fails
        - With it: Controller reduces difficulty, model recovers, tries again later

    Attributes:
        advance_threshold: Accuracy needed to advance (default: 0.9)
        advance_streak: Consecutive successes needed (default: 3)
        backoff_threshold: Loss triggers backoff (default: 2.0)
        ema_decay: Smoothing factor, tunable via curriculum_kwargs (default: 0.9)
        step_size: How much to increase/decrease (default: 5)
    """

    advance_threshold: float = 0.9  # Accuracy to advance
    advance_streak: int = 3  # Consecutive successes needed
    backoff_threshold: float = 2.0  # Loss triggers backoff
    ema_decay: float = 0.9  # Smoothing factor (tunable via curriculum_kwargs)
    step_size: int = 5  # How much to increase/decrease

    def update(self, state: CurriculumState, metrics: Dict[str, float]) -> CurriculumState:
        """Update curriculum based on performance metrics."""
        loss = metrics["loss"]
        accuracy = metrics.get("accuracy", 0.0)

        # Update EMA
        if state.smoothed_accuracy is None:
            smoothed = accuracy
        else:
            smoothed = self.ema_decay * state.smoothed_accuracy + (1 - self.ema_decay) * accuracy

        # Check for advancement
        if smoothed >= self.advance_threshold:
            new_streak = state.consecutive_successes + 1
        else:
            new_streak = 0

        new_bound = state.current_bound

        # Advance if streak reached
        if new_streak >= self.advance_streak:
            new_bound = min(state.max_bound, state.current_bound + self.step_size)
            new_streak = 0  # Reset
            smoothed = 0.0 # Reset EMA on advance

        # CRITICAL: Backoff if diverging at max length
        if loss > self.backoff_threshold and state.current_bound == state.max_bound:
            new_bound = max(state.min_bound, state.current_bound - self.step_size)
            new_streak = 0

        return CurriculumState(
            step=state.step + 1,
            current_bound=new_bound,
            min_bound=state.min_bound,
            max_bound=state.max_bound,
            task_progress=state.task_progress,
            smoothed_accuracy=smoothed,
            consecutive_successes=new_streak,
        )

    def get_params(self, state: CurriculumState) -> Dict[str, Any]:
        """Extract current length from state."""
        return {"length": int(state.current_bound)}


@dataclass
class UniformCurriculum:
    """Uniform random length sampling — matches Delétang et al. 2023.

    Each step randomly samples a new sequence length uniformly from
    [min_bound, max_bound]. Forces the model to generalize across all lengths
    throughout training rather than overfitting to a fixed length distribution.

    Use case: Replicating the paper's UniformCurriculum(values=range(1, 41)).
    """

    def update(self, state: CurriculumState, metrics: Dict[str, float]) -> CurriculumState:
        new_bound = int(np.random.randint(state.min_bound, state.max_bound + 1))
        return CurriculumState(
            step=state.step + 1,
            current_bound=new_bound,
            min_bound=state.min_bound,
            max_bound=state.max_bound,
            task_progress=state.task_progress,
            smoothed_accuracy=state.smoothed_accuracy,
            consecutive_successes=state.consecutive_successes,
        )

    def get_params(self, state: CurriculumState) -> Dict[str, Any]:
        return {"length": int(state.current_bound)}


@dataclass
class MultiTaskCurriculum:
    """Multi-task scheduling with per-task progress tracking.

    Use case: Neural GPU style multi-task training.

    Logic: Maintain per-task progress, schedule based on difficulty.

    Selection strategies:
        - first_unsolved: Pick first task not yet mastered
        - random_weighted: Sample inversely proportional to progress

    Attributes:
        task_names: List of task names to train on
        selection_mode: "first_unsolved" or "random_weighted" (default: "first_unsolved")
        mastery_threshold: Accuracy threshold for mastery (default: 0.95)
        mastered_revisit_ratio: % of steps for mastered tasks to prevent forgetting (default: 0.05)
    """

    task_names: List[str]
    selection_mode: str = "first_unsolved"  # or "random_weighted"
    mastery_threshold: float = 0.95
    mastered_revisit_ratio: float = 0.05  # Configurable: 0 = never revisit

    def update(self, state: CurriculumState, metrics: Dict[str, float]) -> CurriculumState:
        """Update per-task progress based on metrics."""
        task_id = metrics.get("task_id")
        accuracy = metrics.get("accuracy", 0.0)

        # Update progress for current task
        if task_id and state.task_progress is not None:
            new_progress = dict(state.task_progress)
            # EMA update
            old = new_progress.get(task_id, 0.0)
            new_progress[task_id] = 0.9 * old + 0.1 * accuracy
        else:
            new_progress = state.task_progress

        return CurriculumState(
            step=state.step + 1,
            current_bound=state.current_bound,
            min_bound=state.min_bound,
            max_bound=state.max_bound,
            task_progress=new_progress,
            smoothed_accuracy=state.smoothed_accuracy,
            consecutive_successes=state.consecutive_successes,
        )

    def get_params(self, state: CurriculumState) -> Dict[str, Any]:
        """Select next task based on strategy."""
        # Configurable: occasionally revisit mastered tasks to prevent forgetting
        if self.mastered_revisit_ratio > 0 and np.random.random() < self.mastered_revisit_ratio:
            mastered = [t for t in self.task_names if state.task_progress[t] >= self.mastery_threshold]
            if mastered:
                return {"task_id": np.random.choice(mastered)}

        # Select next task
        if self.selection_mode == "first_unsolved":
            for task in self.task_names:
                if state.task_progress[task] < self.mastery_threshold:
                    return {"task_id": task}
            # All mastered, pick hardest
            return {"task_id": min(state.task_progress, key=state.task_progress.get)}

        elif self.selection_mode == "random_weighted":
            # Sample inversely proportional to progress
            weights = [1.0 - state.task_progress[t] for t in self.task_names]
            weights = np.array(weights) / sum(weights)
            task = np.random.choice(self.task_names, p=weights)
            return {"task_id": task}

        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")


def init_curriculum_state(
    strategy: CurriculumStrategy,
    min_bound: int,
    max_bound: int,
    initial_bound: Optional[int] = None,
    task_names: Optional[List[str]] = None,
) -> CurriculumState:
    """Initialize curriculum state from strategy.

    Args:
        strategy: Curriculum strategy instance
        min_bound: Minimum difficulty level
        max_bound: Maximum difficulty level
        initial_bound: Starting difficulty (default: min_bound)
        task_names: List of task names for multi-task curriculum

    Returns:
        Initialized CurriculumState
    """
    if initial_bound is None:
        initial_bound = min_bound

    # Initialize task progress for multi-task
    task_progress = None
    if isinstance(strategy, MultiTaskCurriculum):
        if task_names is None:
            task_names = strategy.task_names
        task_progress = {name: 0.0 for name in task_names}

    return CurriculumState(
        step=0,
        current_bound=initial_bound,
        task_progress=task_progress,
        smoothed_accuracy=None,
        consecutive_successes=0,
        min_bound=min_bound,
        max_bound=max_bound,
    )
