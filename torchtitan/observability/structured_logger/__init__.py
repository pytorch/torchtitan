# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Structured logging: per-rank JSONL trace of training phases.

Typical use::

    from torchtitan.observability import structured_logger as sl

    sl.init_structured_logger(source="training", output_dir="./outputs")
    sl.log_trace_instant("binary_start")
    with sl.log_trace_span("fwd_bwd"):
        ...
"""

from torchtitan.observability.structured_logger.step_state import (
    add_step_tag,
    clear_step_tags,
    get_relative_step,
    get_step,
    get_step_tags,
    set_step,
)
from torchtitan.observability.structured_logger.structured_logging import (
    init_structured_logger,
    log_trace_instant,
    log_trace_scalar,
    log_trace_span,
)

__all__ = [
    "init_structured_logger",
    "set_step",
    "add_step_tag",
    "clear_step_tags",
    "get_step",
    "get_step_tags",
    "get_relative_step",
    "log_trace_scalar",
    "log_trace_instant",
    "log_trace_span",
]
