# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan Observability Library

System metrics: init_observability, set_step, record_span, record_event, EventType
"""

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
]
