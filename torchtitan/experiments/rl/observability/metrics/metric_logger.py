# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MetricLogger + console rendering for typed metrics. See README.md."""

from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from torchtitan.components.metrics import (
    BaseLogger as MetricBackend,
    TensorBoardLogger,
    WandBLogger,
)
from torchtitan.config import Configurable
from torchtitan.tools.console_format import fmt_metric_value
from torchtitan.tools.logging import logger, warn_once
from torchtitan.tools.utils import Color, NoColor

from .types import Metric, MetricValue


__all__ = [
    "MetricBackend",
    "MetricLogger",
    "WandbMetricLogger",
    "log_to_console",
]


# Public name for torchtitan's W&B backend within this package.
WandbMetricLogger = WandBLogger


# ---------------------------------------------------------------------------
# Console rendering (stateless utility, called inside MetricLogger.log).
# ---------------------------------------------------------------------------


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


def log_to_console(
    step: int,
    metrics: dict[str, Any],
    *,
    allow_list: Sequence[str] | None,
    console_prefix: str = "Train",
) -> None:
    """Print one console metric line.

    Example:
        log_to_console(
            step=5,
            metrics={"loss/total": 0.42, "reward/_mean": 1.25},
            allow_list=["loss"],
            console_prefix="Train",
        )
        # Logs: Train | Step:  5  loss/total: 0.42

    Args:
        step: Step number to display.
        metrics: Reduced metrics dict.
        allow_list: Regex search patterns. None prints all keys; [] prints
            nothing; a list prints matching keys in pattern order.
        console_prefix: Text rendered before the step, e.g. "Train" or
            "Validation".
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


# ---------------------------------------------------------------------------
# MetricLogger
# ---------------------------------------------------------------------------


class MetricLogger(Configurable):
    """Aggregates Metric records and dispatches to backends and console.

    Args:
        config: MetricLogger.Config with backend toggles and the
            train/validation console allow lists.
        log_dir: Filesystem directory required when enable_wandb or
            enable_tensorboard is true; backends write under it.
        job_config: Full training job config snapshot, forwarded to
            experiment-tracking backends (W&B) as run metadata.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Metric logger configuration."""

        console_log_keys_train: list[str] | None = field(
            default_factory=lambda: [
                "loss/total",
                "loss/ratio/clipped_frac",
                "reward/_mean",
                "reward/_max",
                "reward/zero_std_frac",
                "rollout/response_length/max",
                "train/grad_norm/mean",
                "train/lr",
                "perf/tokens_per_second",
                "timing/step",
            ]
        )
        """Regex search patterns selecting console keys for train log lines.
        None prints every key; [] prints nothing. Backends always
        receive every reduced metric regardless of this filter."""

        console_log_keys_validation: list[str] | None = field(
            default_factory=lambda: [
                "validation/reward/_mean",
                "validation/reward/_max",
                "validation/response_length/mean",
            ]
        )
        """Same as console_log_keys_train but used when
        MetricLogger.log is called with is_validation=True."""

        enable_wandb: bool = False
        """Log metrics to Weights & Biases."""

        enable_tensorboard: bool = False
        """Log metrics to TensorBoard. Writes under log_dir."""

        wandb_project: str = "titan_rl"
        """W&B project name. Written to WANDB_PROJECT env var via
        setdefault, so a user-set env var takes precedence."""

    def __init__(
        self,
        config: Config,
        *,
        log_dir: str | None = None,
        job_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._backends: list[MetricBackend] = []
        if (config.enable_wandb or config.enable_tensorboard) and log_dir is None:
            raise ValueError(
                "log_dir is required when enable_wandb or enable_tensorboard is True"
            )
        if config.enable_wandb:
            os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
            # Core WandBLogger keeps `config_dict=`; pass through.
            self._backends.append(
                WandbMetricLogger(log_dir=log_dir, config_dict=job_config)
            )
        if config.enable_tensorboard:
            self._backends.append(TensorBoardLogger(log_dir=log_dir))

    @classmethod
    def _aggregate_metrics(cls, metrics: Iterable[Metric]) -> dict[str, float]:
        """Reduces a list of `Metric` records into a single float per metric key.

        Args:
            metrics: Records to aggregate.

        Returns:
            Flat {key: float} dict ready for backend dispatch.

        Example:
            records = [
                Metric("reward", Mean.from_list([0.0, 1.0])),
                Metric("reward", Mean(2)),
                Metric("reward", Max.from_list([0.0, 1.0])),
                Metric("reward", Max(3)),
            ]
            MetricLogger._aggregate_metrics(records)
            # {"reward/mean": 3.0, "reward/max": 3.0}
        """
        groupped_metrics: dict[tuple[str, type], list[MetricValue]] = defaultdict(list)
        for record in metrics:
            groupped_metrics[(record.key, type(record.value))].append(record.value)

        reduced_metrics: dict[str, float] = {}
        for (key, value_cls), values in groupped_metrics.items():
            reduced_outputs = value_cls.reduce(values)
            # A reduction can emit multiple outputs (e.g. SummaryStats).
            for output_suffix, value in reduced_outputs.items():
                output_key = f"{key}/{output_suffix}" if output_suffix else key
                if isinstance(value, float) and math.isnan(value):
                    warn_once(
                        logger,
                        f"Dropping NaN metric {output_key} from "
                        f"{value_cls.__name__} reduction.",
                    )
                    continue
                if output_key in reduced_metrics:
                    raise ValueError(
                        f"Duplicate aggregated metric key {output_key!r}. "
                        "Two reductions expanded to the same output name."
                    )
                reduced_metrics[output_key] = value
        return reduced_metrics

    def log(
        self,
        step: int,
        metrics: Iterable[Metric],
        *,
        is_validation: bool = False,
    ) -> None:
        """Reduce metrics, prints to console, dispatch to backends.

        Console output is filtered by `config.console_log_keys_train` (or
        ..._validation when is_validation=True). Backends always
        receive every reduced key regardless of the filter.

        Args:
            step: Step number to display and pass to backends.
            metrics: Records to aggregate and emit.
            is_validation: When True, use console_log_keys_validation
                and render the prefix "Validation | Step: ...".

        Example:
            metric_logger.log(step=step, metrics=train_metrics)
            metric_logger.log(
                step=step, metrics=validation_metrics, is_validation=True,
            )
        """
        reduced = self._aggregate_metrics(metrics)
        if is_validation:
            allow_list = self.config.console_log_keys_validation
            console_prefix = "Validation"
        else:
            allow_list = self.config.console_log_keys_train
            console_prefix = "Train"
        log_to_console(
            step=step,
            metrics=reduced,
            allow_list=allow_list,
            console_prefix=console_prefix,
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
