# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class Task:
    question: str
    correct_answer: str


class TaskSpec(Protocol):
    """Protocol for pluggable RL tasks.

    Each task defines how to generate prompts, score completions,
    and evaluate correctness.
    """

    def generate_task(self) -> Task: ...

    def get_system_prompt(self) -> str: ...

    def create_eval_instance(self) -> TaskSpec:
        """Return a new instance for evaluation with a different seed."""
        ...

    def reward_function(
        self, completions: list[str], expected_answer: str
    ) -> torch.Tensor:
        """Compute per-completion rewards.

        Args:
            completions: List of completion strings for one prompt.
            expected_answer: Expected answer string.

        Returns:
            Tensor of rewards, one per completion.
        """
        ...

    def compute_step_metrics(self, scored_episodes: list) -> dict[str, str]:
        """Compute task-specific metrics from scored episodes for step logging.

        Returns:
            Dict of metric name to formatted string value.
        """
        ...

    def evaluate_completion(self, text: str, task: Task) -> dict[str, bool]:
        """Evaluate a single completion against a task.

        Returns:
            Dict with at least "correct" and "format_ok" keys.
        """
        ...
