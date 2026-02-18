# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass, field


@dataclass
class Task:
    """A single task instance."""

    question: str
    correct_answer: int
    metadata: dict = field(default_factory=dict)


def extract_answer(text: str) -> int | None:
    """Extract the answer from model output using a fallback chain."""
    # Check for [ANSWER] tag first
    answer_match = re.findall(r"\[ANSWER\]\s*(-?\d+)", text)
    if answer_match:
        return int(answer_match[-1])

    # Try to find explicit answer patterns
    patterns = [
        r"(?:the answer is|answer is|answer:)\s*(-?\d+)",
        r"=\s*(-?\d+)\.?\s*(?:The answer|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return int(matches[-1])

    # Fallback: last number in the text
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None
