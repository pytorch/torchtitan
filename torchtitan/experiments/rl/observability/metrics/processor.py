# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MetricsProcessor for typed metrics. See README.md."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from torchtitan.components.metrics import (
    BaseLogger as MetricBackend,
    TensorBoardLogger,
    WandBLogger,
)
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger, warn_once

from .console import log_to_console
from .types import Metric, MetricValue


__all__ = [
    "MetricBackend",
    "MetricsProcessor",
]


class MetricsProcessor(Configurable):
    """Aggregates Metric records and dispatches to backends and console.

    TODO: unify with torchtitan/components/metrics.py:MetricsProcessor.

    Args:
        config: MetricsProcessor.Config with backend toggles and the
            train/validation console allow lists.
        log_dir: Filesystem directory required when enable_wandb or
            enable_tensorboard is true; backends write under it.
        job_config: Full training job config snapshot, forwarded to
            experiment-tracking backends (W&B) as run metadata.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Metrics processor configuration."""

        console_log_keys_train: list[str] | None = field(
            default_factory=lambda: [
                "loss/mean",
                "loss/ratio/clipped_frac",
                "bit_wise/logprob_diff/max",
                "rollout_reward/_mean",
                "rollout_reward/_max",
                "rollout_reward/zero_std_frac",
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
                # Metric is "validation_reward/_mean" (underscore), like train's
                # "rollout_reward/_mean"; "validation/reward" never matched.
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
                "timing/validate",
            ]
        )
        """Same as console_log_keys_train but used when
        MetricsProcessor.log is called with is_validation=True."""

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
            self._backends.append(WandBLogger(log_dir=log_dir, config_dict=job_config))
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
            MetricsProcessor._aggregate_metrics(records)
            # {"reward/mean": 3.0, "reward/max": 3.0}
        """
        grouped_metrics: dict[tuple[str, type], list[MetricValue]] = defaultdict(list)
        for record in metrics:
            grouped_metrics[(record.key, type(record.value))].append(record.value)

        reduced_metrics: dict[str, float] = {}
        for (key, value_cls), values in grouped_metrics.items():
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
            is_validation: When True, use `console_log_keys_validation`
                and render the prefix "Validation | Step: ...".
                If False, use `console_log_keys_train` and render "Train | Step: ...".

        Example:
            metrics_processor.log(step=step, metrics=train_metrics)
            metrics_processor.log(
                step=step, metrics=validation_metrics, is_validation=True,
            )
        """
        # aggregate metrics
        reduced_metrics = self._aggregate_metrics(metrics)

        # Log to console
        if is_validation:
            allow_list = self.config.console_log_keys_validation
            console_prefix = "Validation"
        else:
            allow_list = self.config.console_log_keys_train
            console_prefix = "Train"
        log_to_console(
            step=step,
            metrics=reduced_metrics,
            allow_list=allow_list,
            console_prefix=console_prefix,
        )

        # log to backends
        for backend in self._backends:
            try:
                backend.log(reduced_metrics, step)
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
