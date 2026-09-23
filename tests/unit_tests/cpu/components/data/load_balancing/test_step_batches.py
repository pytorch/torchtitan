# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan.components.data.types import OptimizerStepLayout


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


def test_optimizer_step_layout_reports_num_microbatches() -> None:
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=3,
    )
    assert layout.num_microbatches == 6
