# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register TorchTitan's DAPO-Math and AIME datasets with Verifiers.

Verifiers 0.3.0 does not provide a v1 taskset for these datasets. This module
also serves as an example of exposing a custom dataset as a Verifiers taskset.
"""

from collections.abc import Iterator
from itertools import islice
from typing import Literal

import verifiers.v1 as vf

from torchtitan.experiments.rl.examples.dapo_math import (
    AIME2025Dataset,
    DapoMathDataset,
    DapoMathSample,
    score_math_response,
)


class VerifiersMathData(vf.TaskData):
    ground_truth: str


class VerifiersMathTask(vf.Task[VerifiersMathData]):
    data: VerifiersMathData

    @vf.reward(weight=1.0)
    async def math_verify(self, trace: vf.Trace) -> float:
        return score_math_response(trace.last_reply or "", self.data.ground_truth)


class VerifiersMathTasksetConfig(vf.TasksetConfig):
    dataset: Literal["dapo_math", "aime2025"] = "dapo_math"


class VerifiersMathTaskset(vf.Taskset[VerifiersMathTask, VerifiersMathTasksetConfig]):
    config: VerifiersMathTasksetConfig

    def load(self) -> list[VerifiersMathTask]:
        dataset, num_tasks = _load_math_dataset(self.config.dataset)
        return [
            VerifiersMathTask(
                VerifiersMathData(
                    idx=index,
                    prompt=sample.prompt,
                    ground_truth=sample.ground_truth,
                ),
                self.config.task,
            )
            for index, sample in enumerate(islice(dataset, num_tasks))
        ]


def _load_math_dataset(
    name: Literal["dapo_math", "aime2025"],
) -> tuple[Iterator[DapoMathSample], int]:
    if name == "dapo_math":
        return DapoMathDataset.Config(shuffle=False).build(), 12643
    return AIME2025Dataset.Config().build(), 30


__all__ = ["VerifiersMathTaskset"]
