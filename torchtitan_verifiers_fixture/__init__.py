# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal Verifiers taskset for integration tests."""

import verifiers.v1 as vf


class DummyData(vf.TaskData):
    pass


class DummyTask(vf.Task[DummyData]):
    @vf.reward(weight=1.0)
    async def nonempty_response(self, trace: vf.Trace) -> float:
        messages = trace.assistant_messages
        content = messages[-1].content if messages else None
        return float(bool(content))


class DummyTaskset(vf.Taskset[DummyTask]):
    def load(self) -> list[DummyTask]:
        return [
            DummyTask(DummyData(idx=index, prompt="Reply with one short sentence."))
            for index in range(5)
        ]


__all__ = ["DummyTaskset"]
