# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Toy MetricsProcessor for the observability experiment."""

from torchtitan.observability import record_event
from torchtitan.observability.step_state import set_step


class MetricsProcessor:
    """Step context manager for the toy trainer.

    Mirrors the method order of components/metrics.py MetricsProcessor
    so the toy and production versions are easy to compare.
    """

    def __init__(self):
        self._step: int = 0

    def set_step(self, step: int) -> None:
        """Set the current training step."""
        self._step = step
        set_step(step)
        record_event({"train.step": step})

    def close(self) -> None:
        """Shutdown. Extended with subprocess cleanup when logging is added."""
        pass
