# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.data.types import (
    OptimizerStepBatch,
    OptimizerStepLayout,
    TrainingMicrobatch,
)


@dataclass(kw_only=True, slots=True)
class _Microbatch(TrainingMicrobatch):
    value: int
    labels: torch.Tensor
    num_valid_tokens: int = 1

    def as_input_dict(self) -> dict[str, Any]:
        return {"value": self.value, "labels": self.labels}


def _microbatch(value: int) -> _Microbatch:
    return _Microbatch(value=value, labels=torch.tensor([value]))


class _ListDataLoader(BaseDataLoader):
    def __init__(self, microbatches: list[TrainingMicrobatch]) -> None:
        self._microbatches = microbatches

    def __iter__(self):
        return iter(self._microbatches)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


@pytest.mark.parametrize(
    ("num_accumulation_steps", "num_pp_microbatches"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_optimizer_step_layout_requires_positive_dimensions(
    num_accumulation_steps: int,
    num_pp_microbatches: int,
) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        OptimizerStepLayout(
            num_accumulation_steps=num_accumulation_steps,
            num_pp_microbatches=num_pp_microbatches,
        )


def test_optimizer_step_layout_groups_flat_microbatches() -> None:
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=3,
    )
    microbatches = [_microbatch(index) for index in range(6)]

    step = layout.group_microbatches(microbatches)

    assert isinstance(step, OptimizerStepBatch)
    assert [
        [microbatch.value for microbatch in group] for group in step.microbatch_groups
    ] == [[0, 1, 2], [3, 4, 5]]


def test_optimizer_step_layout_rejects_wrong_microbatch_count() -> None:
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=2,
    )

    with pytest.raises(ValueError, match="expected 4 microbatches, got 3"):
        layout.group_microbatches([_microbatch(index) for index in range(3)])


def test_optimizer_step_layout_rejects_step_batch_with_wrong_dimensions() -> None:
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=2,
    )

    with pytest.raises(ValueError, match="expected optimizer-step shape 2 x 2"):
        layout.validate_batch(
            OptimizerStepBatch(microbatch_groups=[[_microbatch(0), _microbatch(1)]])
        )


def test_optimizer_step_batch_requires_rectangular_nonempty_groups() -> None:
    with pytest.raises(ValueError, match="at least one microbatch group"):
        OptimizerStepBatch(microbatch_groups=[])

    with pytest.raises(ValueError, match="same positive number"):
        OptimizerStepBatch(
            microbatch_groups=[[_microbatch(0)], [_microbatch(1), _microbatch(2)]]
        )


def test_base_dataloader_groups_one_complete_optimizer_step() -> None:
    loader = _ListDataLoader([_microbatch(index) for index in range(5)])
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=2,
    )

    step = next(loader.iter_optimizer_steps(layout))

    assert [
        [microbatch.value for microbatch in group] for group in step.microbatch_groups
    ] == [[0, 1], [2, 3]]
    assert loader.drain_metrics() == {}


def test_base_dataloader_rejects_partial_optimizer_step() -> None:
    loader = _ListDataLoader([_microbatch(index) for index in range(3)])
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=2,
    )

    with pytest.raises(DataloaderExhaustedError):
        next(loader.iter_optimizer_steps(layout))
