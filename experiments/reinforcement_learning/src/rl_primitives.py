"""
RL primitives for async training.

Provides common dataclasses used across RL training:
- Trajectory: A single rollout from generation
- TrainMetrics: Metrics from a training step
"""

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """A single trajectory from generation."""
    task_question: str
    task_answer: int | str
    response_text: str
    reward: float
    is_correct: bool
    num_turns: int
    num_tool_calls: int
    generator_id: int
    policy_version: int  # Which version of weights was used
    model_only_text: str = ""  # Model-generated text only (no tool results)
    has_answer_tag: bool = False  # Whether model emitted [ANSWER]
    failure_mode: str = ""  # "success", "wrong_format", "tool_spam", "wrong_answer"
    # Pre-tokenized sequence and prompt boundary for training.
    # The generator populates these so the trainer never needs to re-tokenize.
    input_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0  # Number of prompt tokens (response starts here)


@dataclass
class TrainMetrics:
    """Metrics from a training step."""
    step: int
    loss: float
    batch_size: int
    avg_reward: float
    policy_version: int
    correct_rate: float = 0.0  # Fraction with correct answer
    format_rate: float = 0.0  # Fraction that emitted [ANSWER]
