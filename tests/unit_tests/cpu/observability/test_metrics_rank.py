# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan.observability.metrics import _get_metrics_rank


@pytest.mark.parametrize(
    "pp_schedule, expected_rank",
    [
        # Looped schedules: the last stage is on the last pp rank.
        ("1F1B", 4),
        ("Interleaved1F1B", 4),
        # V schedules: the last stage is on pp rank 0.
        ("ZBVZeroBubble", 0),
        ("DualPipeV", 0),
    ],
)
def test_metrics_rank_owns_the_last_stage(pp_schedule: str, expected_rank: int) -> None:
    parallelism_context = SimpleNamespace(pp_enabled=True, world_size=8, pp=2)
    assert (
        _get_metrics_rank(
            parallelism_context=parallelism_context, pp_schedule=pp_schedule
        )
        == expected_rank
    )
