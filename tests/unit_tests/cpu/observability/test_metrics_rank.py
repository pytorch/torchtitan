# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from types import SimpleNamespace

import pytest

from torchtitan.observability.logging import NoColor
from torchtitan.observability.metrics import _get_metrics_rank, ensure_pp_loss_visible


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


@pytest.mark.parametrize(
    "log_rank, expect_warning",
    [
        # No console filter (e.g. scripts/multinode_trainer.slurm): every rank prints.
        (None, False),
        ("", False),
        ("0", True),
        ("0,4", False),
    ],
)
def test_pp_loss_visible_hint_follows_log_rank(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    log_rank: str | None,
    expect_warning: bool,
) -> None:
    if log_rank is None:
        monkeypatch.delenv("LOG_RANK", raising=False)
    else:
        monkeypatch.setenv("LOG_RANK", log_rank)
    parallelism_context = SimpleNamespace(world_size=8, pp=2)
    with caplog.at_level(logging.WARNING, logger="torchtitan.observability.metrics"):
        ensure_pp_loss_visible(
            parallelism_context=parallelism_context,
            pp_schedule="1F1B",
            color=NoColor(),
        )
    assert ("rank 4" in caplog.text) == expect_warning
