# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan Observability Library."""

from torchtitan.observability.aggregation import aggregate, logging_worker
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    MaxMetric,
    MeanMetric,
    MinMetric,
    NoOpMetric,
    record_metric,
    SumMetric,
)
from torchtitan.observability.step_state import add_step_tag, clear_step_tags, set_step
from torchtitan.observability.structured_logging import (
    EventType,
    init_observability,
    record_event,
    record_span,
)

__all__ = [
    "init_observability",
    "set_step",
    "add_step_tag",
    "clear_step_tags",
    "record_span",
    "record_event",
    "EventType",
    "record_metric",
    "MeanMetric",
    "MaxMetric",
    "MinMetric",
    "SumMetric",
    "NoOpMetric",
    "aggregate",
    "logging_worker",
    "EveryNSteps",
]
