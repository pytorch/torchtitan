# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed metric value classes and the keyed Metric envelope. See README.md."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol


__all__ = [
    "Max",
    "Mean",
    "Metric",
    "MetricValue",
    "Min",
    "NoReduce",
    "Std",
    "Sum",
    "SummaryStats",
]


class MetricValue(Protocol):
    """Value object that holds a metric partial value and defines how it
    should be reduced later.

    MetricsProcessor._aggregate_metrics groups `Metric` records by (key, type(MetricValue))
    and calls reduce on each group; the resulting output_suffix becomes
    "<key>/<output_suffix>" in the flat dict sent to backends.

    Subclasses provide `output_suffix`, `from_list`, and `reduce`.

    Example:
        metrics = [
            Metric("reward", Mean.from_list([1.0, 3.0])),  # total=4, count=2
            Metric("reward", Mean(6.0, count=2)),          # total=6, count=2
            Metric("reward", Max.from_list([1.0, 4.0])),
        ]
        # in MetricsProcessor._aggregate_metrics(metrics)
        # {"reward/mean": 2.5, "reward/max": 4.0}
    """

    output_suffix: ClassVar[str]
    """Suffix appended to the metric key when emitted (e.g. 'mean').
    Empty string emits the key unchanged (used by NoReduce)."""

    def __repr__(self) -> str:
        """Human-readable representation of print(self)"""
        kvs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({kvs})"

    @classmethod
    def from_list(cls, values: Sequence[Any]):
        """Build one MetricValue from a list of raw observations.

        Args:
            values: Numeric observations to fold into running stats.

        Example:
            Mean.from_list([1.0, 3.0])  # Mean(value=4.0, count=2)

        Returns:
            A new MetricValue carrying the running totals.
        """
        ...

    @classmethod
    def reduce(cls, metrics: Sequence[object]) -> dict[str, float]:
        """Reduce same-typed MetricValues into output sub-keys.

        Args:
            metrics: Records sharing the same key + same MetricValue subclass.

        Returns:
            {output_suffix: float}. Most subclasses return one entry under
            cls.output_suffix; SummaryStats returns multiple sub-keys.
        """
        ...


@dataclass
class Metric:
    """A metric record that holds a name and a MetricValue payload.

    Args:
        key (str): Metric name to be logged to console and backends (e.g. "reward").
            Note: MetricValue will append output_suffix to the key,
            e.g. "reward" becomes "reward/mean". This means that you
            can define two metrics with the same key but different MetricValue types.
        value (MetricValue): A MetricValue object.

    Example:
        Metric("rollout/response_length", Max.from_list([12, 18, 9]))
        Metric("rollout/response_length", Mean.from_list([12, 18, 9]))
    """

    key: str
    value: MetricValue


class Mean(MetricValue):
    """Weighted mean.

    Say you have two sources of numbers, e.g.:
    source_1 = [1,1,1]
    source_2 = [2]

    If you naively do mean of means, i.e. mean(mean(source_1), mean(source_2)),
    you get mean(1,2) = 1.5, which is incorrect.

    The correct way is to do mean(source_1 + source_2) = (1+1+1+2)/4 = 1.25.

    To avoid holding all numbers in memory, we can hold the sum and count of each.
    source_1 = (value=3, count=3)
    source_2 = (value=2, count=1)
    mean = (3+2)/(3+1) = 1.25

    Args:
        value: Sum of observations (or the single observation when count=1.0).
        count (Optional): Number of observations. Defaults to 1.0.

    Example:
        Mean(value)                              # one obs
        Mean(value, count=N)                     # pre-aggregated
        Mean.from_list(values)                   # many obs
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


class Max(MetricValue):
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
        # Filter NaN records (e.g. an actor that observed nothing) so they
        # don't erase real maxes via Python's NaN comparison semantics.
        finite = [record.value for record in metrics if not math.isnan(record.value)]
        if not finite:
            return {cls.output_suffix: float("nan")}
        return {cls.output_suffix: max(finite)}


class Min(MetricValue):
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
        finite = [record.value for record in metrics if not math.isnan(record.value)]
        if not finite:
            return {cls.output_suffix: float("nan")}
        return {cls.output_suffix: min(finite)}


class Sum(MetricValue):
    """Sum of observed values."""

    output_suffix: ClassVar[str] = "sum"

    def __init__(self, value):
        self.value = float(value)

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Sum:
        return cls(sum(float(value) for value in values))

    @classmethod
    def reduce(cls, metrics: Sequence[Sum]) -> dict[str, float]:
        return {cls.output_suffix: sum(record.value for record in metrics)}


class Std(MetricValue):
    """Population standard deviation (ddof=0).

    Args:
        value: A single observation, or the running total when paired
            with count and sum_squares.
        count: Pre-aggregated observation count. Required iff
            sum_squares is given.
        sum_squares: Pre-aggregated sum of v*v. Required iff count
            is given.

    Example:
        Std(5.0)                                          # one obs
        Std(10.0, count=4, sum_squares=30.0)              # pre-aggregated
        Std.from_list([1.0, 2.0, 3.0, 4.0])               # many obs
    """

    output_suffix: ClassVar[str] = "std"

    def __init__(
        self,
        value,
        *,
        count: float | None = None,
        sum_squares: float | None = None,
    ):
        if (count is None) != (sum_squares is None):
            raise ValueError(
                "Std pre-aggregated input requires both count and sum_squares"
            )
        coerced_value = float(value)
        if count is None:
            self.total = coerced_value
            self.count = 1.0
            self.sum_squares = coerced_value * coerced_value
        else:
            self.total = coerced_value
            self.count = float(count)
            self.sum_squares = float(sum_squares)

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> Std:
        coerced = [float(v) for v in values]
        return cls(
            sum(coerced),
            count=len(coerced),
            sum_squares=sum(v * v for v in coerced),
        )

    @classmethod
    def reduce(cls, metrics: Sequence[Std]) -> dict[str, float]:
        total_count = sum(record.count for record in metrics)
        if total_count == 0:
            return {cls.output_suffix: float("nan")}
        total = sum(record.total for record in metrics)
        total_sum_squares = sum(record.sum_squares for record in metrics)
        mean_value = total / total_count
        # Clamp tiny negatives that arise from floating-point rounding.
        variance = max(0.0, total_sum_squares / total_count - mean_value * mean_value)
        return {cls.output_suffix: math.sqrt(variance)}


class SummaryStats(MetricValue):
    """Population summary statistics: max, mean, min, std, and sum.

    Emits five sub-keys (_max, _mean, _min, _std, _sum)
    with a leading-underscore prefix so the outputs never collide with
    standalone Mean/Max/Min/Std/Sum records under the
    same key.
    """

    output_suffix: ClassVar[str] = ""

    def __init__(self, value):
        coerced_value = float(value)
        self.total = coerced_value
        self.count = 1.0
        self.sum_squares = coerced_value * coerced_value
        self.min_value = coerced_value
        self.max_value = coerced_value

    @classmethod
    def from_list(cls, values: Sequence[Any]) -> SummaryStats:
        coerced = [float(v) for v in values]
        instance = cls.__new__(cls)
        if not coerced:
            instance.total = 0.0
            instance.count = 0.0
            instance.sum_squares = 0.0
            instance.min_value = float("inf")
            instance.max_value = float("-inf")
        else:
            instance.total = sum(coerced)
            instance.count = len(coerced)
            instance.sum_squares = sum(
                observation * observation for observation in coerced
            )
            instance.min_value = min(coerced)
            instance.max_value = max(coerced)
        return instance

    @classmethod
    def reduce(cls, metrics: Sequence[SummaryStats]) -> dict[str, float]:
        total_count = 0.0
        total = 0.0
        total_sum_squares = 0.0
        min_value = float("inf")
        max_value = float("-inf")
        for record in metrics:
            if record.count == 0:
                continue
            total_count += record.count
            total += record.total
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

        mean_value = total / total_count
        variance = max(0.0, total_sum_squares / total_count - mean_value * mean_value)
        return {
            "_max": max_value,
            "_mean": mean_value,
            "_min": min_value,
            "_std": math.sqrt(variance),
            "_sum": total,
        }


class NoReduce(MetricValue):
    """Already-reduced value. Use this as a MetricValue to wrap already reduced
    values. The wrapping exists so that all metrics are processed the same way.

    Example:
        trainer_reduced_metrics: dict[str, float] = trainer.fwd_backward(batch)
        metrics = []
        for key, value in trainer_reduced_metrics.items():
            metrics.append(Metric(key, NoReduce(value)))"""

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
