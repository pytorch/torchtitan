# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MetricLogger + console rendering for typed metrics. See README.md."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from torchtitan.components.metrics import BaseLogger as MetricBackend, WandBLogger
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, NoColor

from .metric_types import aggregate_metrics, Metric


__all__ = [
    "MetricBackend",
    "MetricLogger",
    "WandbMetricLogger",
    "log_to_console",
]


# Public name for torchtitan's W&B backend within this package.
WandbMetricLogger = WandBLogger


# ---------------------------------------------------------------------------
# Console rendering (stateless utility, called inside `MetricLogger.log`).
# ---------------------------------------------------------------------------


# Color cycle for the console output. Skip ``black`` (invisible on dark
# terminals) and ``white`` (low-contrast on light terminals); ``red`` is
# reserved for the ``step:`` prefix.
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
    """Expand ``allow_list`` patterns against ``metrics`` keys.

    Pattern order is preserved; within a pattern, keys sort alphabetically.
    ``None`` returns every key in alphabetical order; ``[]`` returns ``[]``.
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


def _fmt_value(value: Any) -> str:
    """Format a value for console display.

    Examples:

        _fmt_value(3.67060)   -> '3.67'
        _fmt_value(0.00123)   -> '0.0012'
        _fmt_value(1234.5)    -> '1234.5'
        _fmt_value(2e-6)      -> '2.0e-06'
        _fmt_value(True)      -> 'True'
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
    prefix: str = "",
) -> None:
    """Print one console line for ``metrics`` filtered by ``allow_list``.

    For example, ``log_to_console(1, {"reward/mean": 3.67, "my_loss/mean: 0.00123"},
    allow_list=["loss"])`` prints ``step: 1 loss: 0.0012``.

    Column order matches ``allow_list`` pattern order. ``None`` prints
    all keys alphabetically; ``[]`` prints nothing. Missing keys are
    skipped.

    Args:
        step: Step number to display.
        metrics: Reduced metrics dict.
        allow_list: Regex patterns; matching keys print in pattern order.
        prefix: Optional label before "step:" (e.g. "validate ").
    """
    keys = _filter_allow_list(metrics, allow_list)
    if not keys:
        return

    # `isatty` detects if the output is a terminal. If it is, we can use colors.
    color = Color() if sys.stdout.isatty() else NoColor()
    parts = [f"{color.red}{prefix}step: {step:2}"]
    for i, key in enumerate(keys):
        color_name = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        tone = getattr(color, color_name)
        parts.append(f"{tone}{key}: {_fmt_value(metrics[key])}")
    parts.append(color.reset)
    # Single log call so the timestamp prefix only appears once per step.
    # Leading separator visually splits this line from any non-metric log
    # spam above; trailing separator is implicit (next step's leading
    # separator divides them).
    logger.info("%s\n%s", _CONSOLE_SEPARATOR, "  ".join(parts))


# ---------------------------------------------------------------------------
# MetricLogger
# ---------------------------------------------------------------------------


class MetricLogger:
    """Aggregates Metric records and dispatches to metrc logging backends (e.g. WandB)
    + console logging.

    Args:
        backends: Sequence of MetricBackend instances. Each implements
            log(metrics, step) and close().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Config block typically nested as ``RLTrainer.Config.metrics``."""

        log_freq: int = 1
        """Reserved for future cadence control. Values other than 1 currently
        warn; RL still logs every step."""

        train_console_allow_list: list[str] | None = None
        """Regex patterns selecting console keys for **training** steps.
        ``None`` = print all keys (alphabetical). ``[]`` = silent."""

        validation_console_allow_list: list[str] | None = None
        """Regex patterns selecting console keys for **validation** steps.
        ``None`` = print all keys (alphabetical). ``[]`` = silent."""

        enable_wandb: bool = False
        """Log metrics to Weights & Biases."""

        def build(
            self,
            *,
            log_dir: str | None = None,
            config_dict: dict[str, Any] | None = None,
        ) -> MetricLogger:
            return MetricLogger.build(self, log_dir=log_dir, config_dict=config_dict)

    def __init__(self, backends: Sequence[MetricBackend]) -> None:
        self._backends = list(backends)

    def log(
        self,
        step: int,
        metrics: Iterable[Metric],
        *,
        console_allow_list: Sequence[str] | None = None,
        console_prefix: str = "",
    ) -> None:
        """Reduce metrics once, print the console line, dispatch to backends.

        Args:
            step: Step number.
            metrics: Iterable of Metric records.
            console_allow_list: Patterns selecting which metrics should be
                logged to console. None prints all keys alphabetically; []
                suppresses the console line; a regex list prints matching
                keys in pattern order. Same contract as `log_to_console`.
            console_prefix: Optional label before "step:" (e.g. "validate ").
        """
        reduced = aggregate_metrics(metrics)
        log_to_console(
            step=step,
            metrics=reduced,
            allow_list=console_allow_list,
            prefix=console_prefix,
        )
        for backend in self._backends:
            try:
                backend.log(reduced, step)
            except Exception:
                logger.exception(
                    "metric backend %s failed at step %d",
                    type(backend).__name__,
                    step,
                )

    def close(self) -> None:
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                logger.exception(
                    "metric backend %s failed to close", type(backend).__name__
                )

    @classmethod
    def build(
        cls,
        config: "MetricLogger.Config",
        *,
        log_dir: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> MetricLogger:
        """Build a MetricLogger with backends from config.

        Args:
            config: Metrics config (typically from RLTrainer.Config).
            log_dir: Directory passed to wandb.init(dir=...). Required
                when config.enable_wandb=True.
            config_dict: Hyperparameter snapshot for wandb.init(config=...).
                Typically the full job config dict so the W&B UI shows
                every CLI flag that produced this run. Optional.
        """
        if config.log_freq != 1:
            logger.warning(
                "MetricLogger.Config.log_freq=%d is reserved; RL still logs every step.",
                config.log_freq,
            )

        backends: list[MetricBackend] = []
        if config.enable_wandb:
            if log_dir is None:
                raise ValueError(
                    "log_dir is required when MetricLogger.Config.enable_wandb=True"
                )
            # Default the W&B project to ``titan_rl`` so RL runs don't land
            # in the generic torchtitan project. ``setdefault`` keeps any
            # existing ``WANDB_PROJECT`` env var the user already set.
            os.environ.setdefault("WANDB_PROJECT", "titan_rl")
            backends.append(WandbMetricLogger(log_dir=log_dir, config_dict=config_dict))
        return cls(backends)
