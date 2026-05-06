# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed metrics for torchtitan RL. Check README.md for details."""

from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from torchtitan.components.metrics import BaseLogger as MetricBackend, WandBLogger
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, NoColor


__all__ = [
    "Max",
    "Mean",
    "Metric",
    "MetricBackend",
    "MetricLogger",
    "MetricReduction",
    "MetricsConfig",
    "Min",
    "NoReduce",
    "Stats",
    "Std",
    "WandbMetricLogger",
    "aggregate_metrics",
    "log_to_console",
]


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class MetricReduction(Protocol):
    """Protocol every reduction class implements.

    - `from_list(values)` builds a reduction from many raw observations.
    - `reduce(metrics)` combines same-typed records and returns
      `{suffix: float}`.
    - `suffix` is the per-key sub-name the aggregator joins to the metric
      name. `Stats` uses an empty class-level suffix and emits its own
      `_mean`, `_max`, ... sub-keys directly so they don't collide with
      `Mean`/`Max`/`Min`/`Std` under the same metric key.

    Provides a default `__repr__` based on the instance `__dict__`, so
    subclasses inherit pretty printing for free.
    """

    suffix: ClassVar[str]

    def __repr__(self) -> str:
        kvs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({kvs})"

    @classmethod
    def from_list(cls, values: Sequence[Any]):
        ...

    @classmethod
    def reduce(cls, metrics: Sequence[object]) -> dict[str, float]:
        ...


@dataclass
class Metric:
    """A single metric record.

    Args:
        key: Canonical metric name. `/`-separated for nesting.
        reduction: Typed reduction payload.
    """

    key: str
    reduction: MetricReduction


class Mean(MetricReduction):
    """Weighted mean.

    `Mean(value)` records one observation. `Mean(value, count=N)` records
    a pre-aggregated `(sum, count)` pair. `Mean.from_list(values)` records
    many. Combining records is `sum(value) / sum(count)`.
    """

    suffix: ClassVar[str] = "mean"

    def __init__(self, value, count=1.0):
        self.value = float(value)
        self.count = float(count)

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Mean:
        coerced = [float(v) for v in values]
        return cls(value=sum(coerced), count=len(coerced))

    @classmethod
    def reduce(cls, metrics: Sequence[Mean]) -> dict[str, float]:
        total_value = sum(record.value for record in metrics)
        total_count = sum(record.count for record in metrics)
        if total_count == 0:
            return {cls.suffix: float("nan")}
        return {cls.suffix: total_value / total_count}


class Max(MetricReduction):
    """Maximum of observed values. Empty observations return NaN."""

    suffix: ClassVar[str] = "max"

    def __init__(self, value):
        self.value = float(value)

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Max:
        coerced = [float(v) for v in values]
        if not coerced:
            return cls(value=float("nan"))
        return cls(value=max(coerced))

    @classmethod
    def reduce(cls, metrics: Sequence[Max]) -> dict[str, float]:
        if not metrics:
            return {cls.suffix: float("nan")}
        return {cls.suffix: max(record.value for record in metrics)}


class Min(MetricReduction):
    """Minimum of observed values. Empty observations return NaN."""

    suffix: ClassVar[str] = "min"

    def __init__(self, value):
        self.value = float(value)

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Min:
        coerced = [float(v) for v in values]
        if not coerced:
            return cls(value=float("nan"))
        return cls(value=min(coerced))

    @classmethod
    def reduce(cls, metrics: Sequence[Min]) -> dict[str, float]:
        if not metrics:
            return {cls.suffix: float("nan")}
        return {cls.suffix: min(record.value for record in metrics)}


class Std(MetricReduction):
    """Population standard deviation (`ddof=0`).

    Public constructors:
    - `Std(value)` — one observation.
    - `Std.from_list(values)` — many observations.

    Combining records equals (within FP tolerance) the std of the
    concatenation.
    """

    suffix: ClassVar[str] = "std"

    def __init__(self, value):
        coerced_value = float(value)
        self.value = coerced_value
        self.count = 1.0
        self.sum_squares = coerced_value * coerced_value

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Std:
        coerced = [float(v) for v in values]
        instance = cls.__new__(cls)
        instance.value = sum(coerced)
        instance.count = len(coerced)
        instance.sum_squares = sum(observation * observation for observation in coerced)
        return instance

    @classmethod
    def reduce(cls, metrics: Sequence[Std]) -> dict[str, float]:
        total_count = sum(record.count for record in metrics)
        if total_count == 0:
            return {cls.suffix: float("nan")}
        total_value = sum(record.value for record in metrics)
        total_sum_squares = sum(record.sum_squares for record in metrics)
        mean_value = total_value / total_count
        # Clamp tiny negatives that arise from floating-point rounding.
        variance = max(0.0, total_sum_squares / total_count - mean_value * mean_value)
        return {cls.suffix: math.sqrt(variance)}


class Stats(MetricReduction):
    """Population summary stats.

    Emits 5 sub-keys with a leading-underscore prefix so they never
    collide with `Mean`/`Max`/`Min`/`Std` under the same metric key:
    `_max`, `_mean`, `_min`, `_std`, `_sum`. Output keys read like
    ``reward/_mean``, ``reward/_max``, ...

    Public constructors:
    - `Stats(value)` — one observation.
    - `Stats.from_list(values)` — many observations.
    """

    suffix: ClassVar[str] = ""

    def __init__(self, value):
        coerced_value = float(value)
        self.value = coerced_value
        self.count = 1.0
        self.sum_squares = coerced_value * coerced_value
        self.min_value = coerced_value
        self.max_value = coerced_value

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Stats:
        coerced = [float(v) for v in values]
        instance = cls.__new__(cls)
        if not coerced:
            instance.value = 0.0
            instance.count = 0.0
            instance.sum_squares = 0.0
            instance.min_value = float("inf")
            instance.max_value = float("-inf")
        else:
            instance.value = sum(coerced)
            instance.count = len(coerced)
            instance.sum_squares = sum(
                observation * observation for observation in coerced
            )
            instance.min_value = min(coerced)
            instance.max_value = max(coerced)
        return instance

    @classmethod
    def reduce(cls, metrics: Sequence[Stats]) -> dict[str, float]:
        total_count = 0.0
        total_value = 0.0
        total_sum_squares = 0.0
        min_value = float("inf")
        max_value = float("-inf")
        for record in metrics:
            if record.count == 0:
                continue
            total_count += record.count
            total_value += record.value
            total_sum_squares += record.sum_squares
            min_value = min(record.min_value, min_value)
            max_value = max(record.max_value, max_value)

        if total_count == 0:
            return {
                "_max": float("nan"),
                "_mean": float("nan"),
                "_min": float("nan"),
                "_std": float("nan"),
                "_sum": 0.0,
            }

        mean_value = total_value / total_count
        variance = max(0.0, total_sum_squares / total_count - mean_value * mean_value)
        return {
            "_max": max_value,
            "_mean": mean_value,
            "_min": min_value,
            "_std": math.sqrt(variance),
            "_sum": total_value,
        }


class NoReduce(MetricReduction):
    """Already-reduced value; logged unchanged.

    Use to wrap values reduced upstream of this module (e.g. scalars
    returned by an actor that already ran `dist.all_reduce`). Empty
    suffix so `Metric("loss", NoReduce(0.5))` logs as `loss`.
    """

    suffix: ClassVar[str] = ""

    def __init__(self, value):
        self.value = float(value)

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> NoReduce:
        if len(values) != 1:
            raise ValueError(
                f"NoReduce.from_list expects exactly one value; got {len(values)}."
            )
        return cls(value=float(values[0]))

    @classmethod
    def reduce(cls, metrics: Sequence[NoReduce]) -> dict[str, float]:
        if len(metrics) != 1:
            raise ValueError(
                f"NoReduce expects exactly one entry per key; got {len(metrics)}."
            )
        return {cls.suffix: metrics[0].value}


def aggregate_metrics(metrics: Iterable[Metric]) -> dict[str, float]:
    """Group `Metric` records by `(key, reduction type)` and reduce.

    Each reduction's `reduce` returns `{suffix: value}`; the aggregator
    joins `f"{key}/{suffix}" if suffix else key`. NaN entries are
    filtered out so empty inputs simply do not appear in the output.

    Raises:
        ValueError: If two reductions write the same output key.
    """
    # Group by {(key, reduction type): [reductions]}.
    groups: dict[tuple[str, type], list[MetricReduction]] = defaultdict(list)
    for record in metrics:
        groups[(record.key, type(record.reduction))].append(record.reduction)

    # Apply reduce to each group.
    output: dict[str, float] = {}
    for (key, reduction_cls), reductions in groups.items():
        # Reduce op returns a dict of `{suffix: value}`, e.g. {"max": 1.67}
        # A reduce type may emit multiple keys,
        # e.g. {"stats_max": 1.67, "stats_min": 0.0}
        reduced_subkeys = reduction_cls.reduce(reductions)
        for sub_suffix, value in reduced_subkeys.items():
            # NaN entries are filtered out.
            # Happens if empty list is logged using .from_list([]).
            if isinstance(value, float) and math.isnan(value):
                continue
            output_key = f"{key}/{sub_suffix}" if sub_suffix else key
            if output_key in output:
                raise ValueError(
                    f"Duplicate aggregated metric key {output_key!r}. "
                    "Two reductions expanded to the same output name."
                )
            output[output_key] = value
    return output


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


# Public name for torchtitan's W&B backend within this package. Same class
# as `torchtitan.components.metrics.WandBLogger`.
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


def _expand_keys(
    aggregated: dict[str, Any],
    allow_list: Sequence[str] | None,
) -> list[str]:
    """Expand ``allow_list`` patterns against ``aggregated`` keys.

    Pattern order is preserved; within a pattern, keys sort alphabetically.
    ``None`` returns every key in alphabetical order; ``[]`` returns ``[]``.
    """
    if allow_list is None:
        return sorted(aggregated)
    seen: list[str] = []
    seen_set: set[str] = set()
    for raw in allow_list:
        pattern = re.compile(raw)
        matches = sorted(
            key for key in aggregated if pattern.search(key) and key not in seen_set
        )
        for key in matches:
            seen.append(key)
            seen_set.add(key)
    return seen


def _fmt_value(value: Any) -> str:
    """Format a value for console display.

    Numbers: show at least 2 non-zero decimals, up to 5 decimal places.
    Values smaller than ``1e-5`` (no significant digit fits in 5
    decimals) fall back to scientific notation so things like
    ``train/lr=2e-6`` don't render as ``0.00000``.
    Other types (bool, str, ...) use `str(value)` as-is.

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
    # No significant digit fit in 5 decimals — fall back to scientific
    # so very small values (e.g. ``2e-6`` learning rates) still convey
    # a magnitude instead of rendering as ``0.00000``.
    if n_nonzero == 0:
        return f"{value:.1e}"
    return f"{value:.{max(n_decimals, 2)}f}"


def log_to_console(
    step: int,
    aggregated: dict[str, Any],
    *,
    allow_list: Sequence[str] | None,
    prefix: str = "",
) -> None:
    """Render one ``step: N | k1: v1 | ...`` line to the standard logger.

    Stateless — each call expands ``allow_list`` against ``aggregated``,
    so the column order is deterministic per allow-list and only keys
    actually present this step are rendered (no ``--`` placeholders).

    Args:
        step: Training step (or validation step) number.
        aggregated: Reduced metrics dict (e.g. output of
            ``aggregate_metrics``). Keys not in ``allow_list`` are dropped.
        allow_list: Regex patterns (matched via ``re.search``); only
            matching keys are printed, in pattern order. ``None`` prints
            every key in alphabetical order. ``[]`` prints nothing.
        prefix: Optional label printed before ``step:`` (e.g.
            ``"validate "``).
    """
    keys = _expand_keys(aggregated, allow_list)
    if not keys:
        return
    color = Color() if sys.stdout.isatty() else NoColor()
    parts = [f"{color.red}{prefix}step: {step:2}"]
    for i, key in enumerate(keys):
        color_name = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        tone = getattr(color, color_name)
        parts.append(f"{tone}{key}: {_fmt_value(aggregated[key])}")
    parts.append(color.reset)
    logger.info(_CONSOLE_SEPARATOR)
    logger.info("  ".join(parts))
    logger.info(_CONSOLE_SEPARATOR)


# ---------------------------------------------------------------------------
# Config + Logger
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class MetricsConfig:
    """Config block exposed on `RLTrainer.Config` as `metrics:`.

    Matches torchtitan's pretraining convention (`MetricsProcessor.Config`)
    while staying RL-specific.
    """

    log_freq: int = 1
    """Log every N training steps. Currently a no-op — RL logs every
    step. Kept on the surface for forward compatibility; see
    ``IDEAS.md`` for the trainer-side cost-reduction follow-up.

    TODO(rl-metrics-log-freq): honor this on the trainer-side reduction
    path (skip the loss+verification all_reduce on non-log steps) so
    `log_freq>1` actually saves work, not just bytes.
    """

    train_console_allow_list: list[str] | None = None
    """Regex patterns selecting console keys for **training** steps.
    ``None`` = print all keys (alphabetical). ``[]`` = silent."""

    validation_console_allow_list: list[str] | None = None
    """Regex patterns selecting console keys for **validation** steps.
    ``None`` = print all keys (alphabetical). ``[]`` = silent."""

    enable_wandb: bool = False
    """Log metrics to Weights & Biases."""


class MetricLogger:
    """Aggregates a ``list[Metric]`` once per ``log`` call, renders the
    console line in-place, and fans the reduced dict out to every
    backend (W&B, ...).

    Console output is intentionally **not** a backend: callers pass a
    per-call ``console_allow_list`` to render a context-specific line
    (training vs. validation) without re-aggregating.

    Args:
        backends: Sequence of `MetricBackend` instances. Each implements
            `log(metrics, step)` and `close()`.
    """

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
        """Reduce ``metrics`` once, render the console line, and dispatch
        to every backend.

        Args:
            step: Training step (or validation step) number.
            metrics: Iterable of `Metric` records.
            console_allow_list: Patterns selecting console keys for this
                call. ``None`` (default) prints every key alphabetically;
                ``[]`` suppresses the console render entirely; a list of
                regex patterns prints matching keys in pattern order.
                Same contract as :func:`log_to_console`.
            console_prefix: Optional label printed before ``step:``
                (e.g. ``"validate "``).
        """
        reduced = aggregate_metrics(metrics)
        log_to_console(
            step,
            reduced,
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
        config: MetricsConfig,
        *,
        log_dir: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> MetricLogger:
        """Build a `MetricLogger` with default backends from `config`.

        Console rendering happens inline in :meth:`log` and isn't a
        backend; ``config``'s ``*_console_allow_list`` fields are read
        by the controller, not by this method.

        Args:
            config: Metrics config block (typically from `RLTrainer.Config`).
            log_dir: Directory passed to `wandb.init(dir=...)`. Required
                when `config.enable_wandb=True`.
            config_dict: Hyperparameter snapshot passed to
                `wandb.init(config=...)`. Typically the full job config dict
                (e.g. `config.to_dict()`) so the W&B UI shows every CLI flag
                that produced this run. Optional.
        """
        if config.log_freq != 1:
            # See ``MetricsConfig.log_freq`` docstring + ``IDEAS.md``;
            # warn so users don't silently expect a no-op to take effect.
            logger.warning(
                "MetricsConfig.log_freq=%d is currently a no-op; RL logs "
                "every step. Tracked as TODO(rl-metrics-log-freq).",
                config.log_freq,
            )

        backends: list[MetricBackend] = []
        if config.enable_wandb:
            if log_dir is None:
                raise ValueError(
                    "log_dir is required when MetricsConfig.enable_wandb=True"
                )
            # Default the W&B project to ``titan_rl`` so RL runs don't land
            # in the generic torchtitan project. ``setdefault`` keeps any
            # existing ``WANDB_PROJECT`` env var the user already set.
            os.environ.setdefault("WANDB_PROJECT", "titan_rl")
            backends.append(
                WandbMetricLogger(
                    log_dir=log_dir,
                    config_dict=config_dict,
                )
            )
        return cls(backends)
