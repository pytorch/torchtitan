# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-step context: current step number and step tags."""

_STEP: int | None = None
_STEP_TAGS: tuple[str, ...] = ()


def set_step(step: int) -> None:
    """Set the current training step. All subsequent JSONL records will
    include this step number. Clears step tags from the previous step.

    Example:

        for step in range(1, num_steps + 1):
            set_step(step)
            train_step(...)
    """
    global _STEP, _STEP_TAGS
    _STEP = step
    _STEP_TAGS = ()


def get_step() -> int | None:
    """Return the current step, or None if not set."""
    return _STEP


def get_step_tags() -> tuple[str, ...]:
    """Return the current step tags."""
    return _STEP_TAGS


def add_step_tag(tag: str) -> None:
    """Annotate the current step. Tags appear in system JSONL for filtering.

    Example:

        if gc_happened:
            add_step_tag("gc")
        if is_validation:
            add_step_tag("eval")
        # system JSONL: {"normvector": {"step_tags": ["gc", "eval"]}}
    """
    global _STEP_TAGS
    if tag not in _STEP_TAGS:
        _STEP_TAGS = _STEP_TAGS + (tag,)


def clear_step_tags() -> None:
    """Reset step tags."""
    global _STEP_TAGS
    _STEP_TAGS = ()
