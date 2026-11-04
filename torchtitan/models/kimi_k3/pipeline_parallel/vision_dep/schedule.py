# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline schedule with the decoupled encoder process around each step."""

from __future__ import annotations

from typing import Any

from torch.distributed.pipelining.schedules import _PipelineSchedule

from .runtime import VisionDep


class VisionDepSchedule(_PipelineSchedule):
    """The pipeline schedule with the vision phases around each training step."""

    def __init__(self, schedule: _PipelineSchedule, dep: VisionDep) -> None:
        self._schedule = schedule
        self._dep = dep

    def __getattr__(self, name: str) -> Any:
        if name == "_schedule":
            raise AttributeError(name)
        return getattr(self._schedule, name)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        kwarg_mbs = kwargs.get("kwarg_mbs")
        if kwarg_mbs is None:
            raise ValueError("vision_dep plans a step from its kwarg_mbs.")
        self._dep.begin_step(kwarg_mbs)
        try:
            output = self._schedule.step(*args, **kwargs)
        except BaseException:
            self._dep.abort_step()
            raise
        self._dep.end_step()
        return output

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        return self._schedule.eval(*args, **kwargs)

    def _step_microbatches(self, *args: Any, **kwargs: Any) -> Any:
        return self._schedule._step_microbatches(*args, **kwargs)
