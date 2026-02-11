# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared utilities for loss extraction and comparison.

This module provides common functionality used by both:
- scripts/loss_compare.py (CLI tool for comparing losses across commits)
- tests/integration_tests/run_tests.py (integration test runner)
"""

import re


def extract_losses_from_log(log_file: str) -> dict[int, float]:
    """Extract step and loss pairs from a training log file.

    Parses log lines matching the pattern: "step: N loss: X.XXXX"
    Handles ANSI escape codes that may be present in colored terminal output.

    Args:
        log_file: Path to the training log file

    Returns:
        Dictionary mapping step numbers to loss values
    """
    losses = {}
    step_loss_pattern = re.compile(r"step:\s*(\d+)\s*loss:\s*(\d+\.\d+)")
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    with open(log_file, "r") as f:
        for line in f:
            # Strip ANSI codes before matching
            clean_line = ansi_escape.sub("", line)
            match = step_loss_pattern.search(clean_line)
            if match:
                step, loss = match.groups()
                losses[int(step)] = float(loss)

    return losses


def compare_losses(
    losses1: dict[int, float],
    losses2: dict[int, float],
    name1: str = "run1",
    name2: str = "run2",
) -> tuple[bool, str]:
    """Compare two loss dictionaries for equality.

    Args:
        losses1: First loss dictionary (step -> loss)
        losses2: Second loss dictionary (step -> loss)
        name1: Name for first run (for error messages)
        name2: Name for second run (for error messages)

    Returns:
        Tuple of (success: bool, message: str)
        - success is True if all losses match exactly
        - message contains details about the comparison or mismatch
    """
    if not losses1:
        return False, f"No losses found in {name1}"

    if not losses2:
        return False, f"No losses found in {name2}"

    steps1 = set(losses1.keys())
    steps2 = set(losses2.keys())

    if steps1 != steps2:
        return False, (
            f"Steps mismatch: {name1} has {len(steps1)} steps, "
            f"{name2} has {len(steps2)} steps"
        )

    mismatches = []
    for step in sorted(steps1):
        loss1 = losses1[step]
        loss2 = losses2[step]
        if loss1 != loss2:
            mismatches.append(f"  step {step}: {name1}={loss1}, {name2}={loss2}")

    if mismatches:
        return False, "Loss mismatches:\n" + "\n".join(mismatches)

    return True, f"All {len(steps1)} steps have identical losses"
