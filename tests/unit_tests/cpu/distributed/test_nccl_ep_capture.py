# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast, TYPE_CHECKING

import pytest
from torchtitan.distributed.nccl_ep_capture import (
    get_buffer_owner,
    NCCLEPBufferOwner,
    NCCLEPRoutingCapturePlan,
    routing_capture_plan,
)

if TYPE_CHECKING:
    from torchtitan.distributed.nccl_ep import RoutingSlots


def test_buffer_owner_requires_scope_and_closes_initialized_dispatchers_once():
    events = []
    owner = NCCLEPBufferOwner()
    with pytest.raises(ValueError, match="explicit native-buffer owner"):
        get_buffer_owner()
    with owner.initialize():
        assert get_buffer_owner() is owner
        owner.dispatchers.append(
            SimpleNamespace(close_buffer=lambda: events.append("close"))
        )
    with pytest.raises(ValueError, match="explicit native-buffer owner"):
        get_buffer_owner()
    owner.close()
    owner.close()
    assert events == ["close"]


def test_capture_failure_releases_partial_leases_and_resets_context():
    plan = NCCLEPRoutingCapturePlan()
    released = []
    pool = SimpleNamespace(
        generations=[4, 2],
        slots=[[object(), True, True], [object(), True, True]],
        release=lambda index, generation: released.append((index, generation)),
    )
    from threading import RLock

    pool.lock = RLock()

    # A graph captured by an earlier call stays live and keeps its lease.
    plan.captured.append((cast("RoutingSlots", pool), 1, 2))

    def fail():
        assert routing_capture_plan.get() is plan
        plan.captured.append((cast("RoutingSlots", pool), 0, 4))
        raise RuntimeError("failed capture")

    with pytest.raises(RuntimeError, match="failed capture"):
        plan.invoke(plan.run, fail)
    assert routing_capture_plan.get() is None
    assert released == [(0, 4)]
    assert not pool.slots[0][2] and pool.slots[1][2]
    assert not plan.prepared
    assert plan.captured == [(pool, 1, 2)]


def test_prepare_runs_after_observed_warmup_without_hidden_forward():
    plan = NCCLEPRoutingCapturePlan()
    events = []

    def forward(value):
        assert routing_capture_plan.get() is plan
        events.append(value)
        return value

    assert plan.run(forward, "warmup") == "warmup"
    assert not plan.prepared
    plan.prepare_for_capture("capture_input")
    assert plan.prepared
    assert plan.run(forward, "capture") == "capture"
    assert events == ["warmup", "capture"]
    plan.release()
    assert not plan.prepared
