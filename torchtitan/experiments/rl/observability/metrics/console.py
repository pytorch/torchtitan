# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics console logging for typed metrics. See README.md.

TODO: unify these console-rendering utilities with torchtitan's main
trainer console logging in torchtitan/components/metrics.py.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from typing import Any

from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, NoColor

# Color cycle for the console output. Skip `black` (invisible on dark
# terminals) and `white` (low-contrast on light terminals); `red` is
# reserved for the leading "{prefix} | Step:" rendering.
_COLOR_CYCLE = [
    name
    for name in vars(Color)
    if not name.startswith("_") and name not in ("reset", "black", "white", "red")
]


# Visual separator framing the metric line so it stands out from
# surrounding actor / framework log spam.
_CONSOLE_SEPARATOR = "-" * 10


def _filter_allow_list(
    metrics: dict[str, Any],
    allow_list: Sequence[str] | None,
) -> list[str]:
    """Expand allow_list patterns against metrics keys.

    None returns every key in alphabetical order; [] returns [].
    """
    if allow_list is None:
        return sorted(metrics)
    seen: list[str] = []
    seen_set: set[str] = set()
    for raw in allow_list:
        pattern = re.compile(raw)
        matches = sorted(
            key for key in metrics if pattern.search(key) and key not in seen_set
        )
        for key in matches:
            seen.append(key)
            seen_set.add(key)
    return seen


def fmt_metric_value(value: Any) -> str:
    """Format a value for console display.

    Examples:

        fmt_metric_value(3.67060)   -> '3.67'
        fmt_metric_value(0.00123)   -> '0.0012'
        fmt_metric_value(1234.5)    -> '1234.5'
        fmt_metric_value(2e-6)      -> '2.0e-06'
        fmt_metric_value(True)      -> 'True'
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if value == 0:
        return "0"
    if isinstance(value, int) or abs(value) >= 100:
        return f"{value:.1f}"

    frac = abs(value) - int(abs(value))
    if frac == 0:
        return f"{value:.1f}"

    # Walk decimal digits until we've seen two non-zero ones, capped at 5.
    n_decimals, n_nonzero = 0, 0
    temp = frac
    while n_decimals < 5 and n_nonzero < 2:
        n_decimals += 1
        temp *= 10
        if int(temp) % 10 != 0 or n_nonzero > 0:
            n_nonzero += 1
    if n_nonzero == 0:
        return f"{value:.1e}"
    return f"{value:.{max(n_decimals, 2)}f}"


def log_to_console(
    step: int,
    metrics: dict[str, Any],
    *,
    allow_list: Sequence[str] | None,
    console_prefix: str = "Train",
) -> None:
    """Print one console metric line.

    Args:
        step: Step number to display.
        metrics: Reduced metrics dict.
        allow_list: Regex search patterns. None prints all keys; [] prints
            nothing; a list prints matching keys in pattern order.
        console_prefix: Text rendered before the step, e.g. "Train" or
            "Validation".

    Example:
        log_to_console(
            step=5,
            metrics={"loss/mean": 0.42, "reward/_mean": 1.25},
            allow_list=["loss"],
            console_prefix="Train",
        )
        # Logs: Train | Step:  5  loss/mean: 0.42
    """
    keys = _filter_allow_list(metrics, allow_list)
    if not keys:
        return

    # `isatty` detects if the output is a terminal. If it is, we can use colors.
    color = Color() if sys.stdout.isatty() else NoColor()
    parts = [f"{color.red}{console_prefix} | Step: {step:2}"]
    for i, key in enumerate(keys):
        color_name = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        tone = getattr(color, color_name)
        parts.append(f"{tone}{key}: {fmt_metric_value(metrics[key])}")
    parts.append(color.reset)
    # Single log call so the timestamp prefix only appears once per step.
    # Leading separator visually splits this line from other logs.
    logger.info("%s\n%s", _CONSOLE_SEPARATOR, "  ".join(parts))
