# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed metric reductions and the aggregator. Check README.md for details."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol


__all__ = [
    "Max",
    "Mean",
    "Metric",
    "MetricReduction",
    "Min",
    "NoReduce",
    "Stats",
    "Std",
    "aggregate_metrics",
]


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class MetricReduction(Protocol):
    """Protocol every reduction class implements.

    - `from_list(values)` builds a reduction from many raw observations.
    - `reduce(metrics)` combines same-typed records and returns
      `{output_suffix: float}`.
    - `output_suffix` is the per-key sub-name the aggregator joins to the metric
      name. E.g. a metric "reward" can add a "mean" suffix to their key output
      if they so choose.

    Provides a default `__repr__` so subclasses inherit pretty printing for free.
    """

    output_suffix: ClassVar[str]

    def __repr__(self) -> str:
        kvs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({kvs})"

    @classmethod
    def from_list(cls, values: Sequence[Any]): ...

    @classmethod
    def reduce(cls, metrics: Sequence[object]) -> dict[str, float]: ...


@dataclass
class Metric:
    """A single metric record.

    Args:
        key: Canonical metric name. ``/``-separated for nesting.
        reduction: Typed reduction payload.
    """

    key: str
    reduction: MetricReduction


class Mean(MetricReduction):
    """Weighted mean.

    ``Mean(value)`` records one observation. ``Mean(value, count=N)``
    records a pre-aggregated ``(sum, count)`` pair. ``Mean.from_list(values)``
    records many. Combining records is ``sum(value) / sum(count)``.
    """

    output_suffix: ClassVar[str] = "mean"

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
            return {cls.output_suffix: float("nan")}
        return {cls.output_suffix: total_value / total_count}


class Max(MetricReduction):
    """Maximum of observed values. Empty observations return NaN."""

    output_suffix: ClassVar[str] = "max"

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
            return {cls.output_suffix: float("nan")}
        return {cls.output_suffix: max(record.value for record in metrics)}


class Min(MetricReduction):
    """Minimum of observed values. Empty observations return NaN."""

    output_suffix: ClassVar[str] = "min"

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
            return {cls.output_suffix: float("nan")}
        return {cls.output_suffix: min(record.value for record in metrics)}


class Std(MetricReduction):
    """Population standard deviation (``ddof=0``).

    Public constructors:
    - ``Std(value)`` - one observation.
    - ``Std.from_list(values)`` - many observations.

    Combining records equals (within FP tolerance) the std of the
    concatenation.
    """

    output_suffix: ClassVar[str] = "std"

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
            return {cls.output_suffix: float("nan")}
        total_value = sum(record.value for record in metrics)
        total_sum_squares = sum(record.sum_squares for record in metrics)
        mean_value = total_value / total_count
        # Clamp tiny negatives that arise from floating-point rounding.
        variance = max(0.0, total_sum_squares / total_count - mean_value * mean_value)
        return {cls.output_suffix: math.sqrt(variance)}


class Stats(MetricReduction):
    """Population summary stats.

    Emits 5 sub-keys with a leading-underscore prefix so they never
    collide with ``Mean``/``Max``/``Min``/``Std`` under the same metric
    key: ``_max``, ``_mean``, ``_min``, ``_std``, ``_sum``. Output keys
    read like ``reward/_mean``, ``reward/_max``, ...

    Public constructors:
    - ``Stats(value)`` - one observation.
    - ``Stats.from_list(values)`` - many observations.
    """

    output_suffix: ClassVar[str] = ""

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
    returned by an actor that already ran ``dist.all_reduce``). Empty
    suffix so ``Metric("loss", NoReduce(0.5))`` logs as ``loss``.
    """

    output_suffix: ClassVar[str] = ""

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
        return {cls.output_suffix: metrics[0].value}


def aggregate_metrics(metrics: Iterable[Metric]) -> dict[str, float]:
    """Group ``Metric`` records by ``(key, reduction type)`` and reduce.

    Each reduction's ``reduce`` returns ``{output_suffix: value}``; the
    aggregator joins ``f"{key}/{output_suffix}" if output_suffix else key``.
    NaN entries are filtered out so empty inputs simply do not appear in
    the output.

    Raises:
        ValueError: If two reductions write the same output key.
    """
    # Group by {(key, reduction type): [reductions]}.
    groups: dict[tuple[str, type], list[MetricReduction]] = defaultdict(list)
    for record in metrics:
        groups[(record.key, type(record.reduction))].append(record.reduction)

    output: dict[str, float] = {}
    for (key, reduction_cls), reductions in groups.items():
        # Reduce returns {output_suffix: value}, e.g. {"max": 1.67}.
        # Multi-output reductions emit several entries (e.g. Stats).
        reduced_outputs = reduction_cls.reduce(reductions)
        for output_suffix, value in reduced_outputs.items():
            # NaN entries are filtered out.
            if isinstance(value, float) and math.isnan(value):
                continue
            output_key = f"{key}/{output_suffix}" if output_suffix else key
            if output_key in output:
                raise ValueError(
                    f"Duplicate aggregated metric key {output_key!r}. "
                    "Two reductions expanded to the same output name."
                )
            output[output_key] = value
    return output
