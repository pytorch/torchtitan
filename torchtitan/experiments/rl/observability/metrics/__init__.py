# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed metrics for torchtitan RL. Check README.md for details.

Public surface - import as::

    from torchtitan.experiments.rl.observability import metrics as m

and use m.Mean, m.MetricLogger, etc. Aggregation is on the logger:
m.MetricLogger.aggregate_metrics(records).
"""

from .metric_logger import (
    log_to_console,
    MetricBackend,
    MetricLogger,
    WandbMetricLogger,
)
from .types import Max, Mean, Metric, MetricValue, Min, NoReduce, Std, Sum, SummaryStats


# Convenience alias matching torchtitan's pretraining `MetricsProcessor.Config`
# spelling. Equivalent to `MetricLogger.Config`.
MetricsConfig = MetricLogger.Config


__all__ = [
    "Max",
    "Mean",
    "Metric",
    "MetricBackend",
    "MetricLogger",
    "MetricsConfig",
    "MetricValue",
    "Min",
    "NoReduce",
    "Std",
    "Sum",
    "SummaryStats",
    "WandbMetricLogger",
    "log_to_console",
]
