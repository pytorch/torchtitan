# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed metrics for torchtitan RL. Check README.md for details.

Public surface - import as::

    from torchtitan.experiments.rl.observability import metrics as m

and use ``m.Mean``, ``m.MetricLogger``, ``m.aggregate_metrics``, etc.
"""

from .metric_logger import (
    log_to_console,
    MetricBackend,
    MetricLogger,
    WandbMetricLogger,
)
from .metric_types import (
    aggregate_metrics,
    Max,
    Mean,
    Metric,
    MetricReduction,
    Min,
    NoReduce,
    Stats,
    Std,
)


# Convenience alias matching torchtitan's pretraining `MetricsProcessor.Config`
# spelling. Equivalent to `MetricLogger.Config`.
MetricsConfig = MetricLogger.Config


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
