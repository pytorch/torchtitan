# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the in-mesh DP request router (`IntraGeneratorRouter`)."""

from __future__ import annotations

import pytest

from torchtitan.experiments.rl.routing.intra_generator_router import (
    IntraGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)


def _loads(router: IntraGeneratorRouter) -> list[int]:
    return [h.reserved_load for h in router._handles]


def test_round_robin_cycles_across_dp_ranks():
    router = IntraGeneratorRouter.Config(
        strategy=RoundRobinRoutingStrategy.Config()
    ).build(dp_degree=3)
    chosen = [router.reserve(f"r{i}", routing_session_id=None) for i in range(7)]
    assert chosen == [0, 1, 2, 0, 1, 2, 0]


def test_least_loaded_balances_in_flight_count():
    router = IntraGeneratorRouter.Config(
        strategy=LeastLoadedRoutingStrategy.Config()
    ).build(dp_degree=2)
    # Each reserve adds one load unit; idle ties resolve to the lowest index.
    assert router.reserve("r0", routing_session_id=None) == 0
    assert router.reserve("r1", routing_session_id=None) == 1
    # Both ranks now hold one request; the tie again resolves to rank 0.
    assert router.reserve("r2", routing_session_id=None) == 0
    assert _loads(router) == [2, 1]
    # Freeing rank 0's requests makes it the least loaded again.
    router.release("r0")
    router.release("r2")
    assert router.reserve("r3", routing_session_id=None) == 0


def test_release_frees_load():
    router = IntraGeneratorRouter.Config(
        strategy=LeastLoadedRoutingStrategy.Config()
    ).build(dp_degree=2)
    router.reserve("r0", routing_session_id=None)
    router.reserve("r1", routing_session_id=None)
    router.release("r0")
    router.release("r1")
    assert _loads(router) == [0, 0]
    assert router._reservations == {}


def test_duplicate_reserve_asserts():
    router = IntraGeneratorRouter.Config(
        strategy=LeastLoadedRoutingStrategy.Config()
    ).build(dp_degree=2)
    router.reserve("r0", routing_session_id=None)
    with pytest.raises(AssertionError, match="already has a reservation"):
        router.reserve("r0", routing_session_id=None)


def test_release_unknown_request_raises():
    router = IntraGeneratorRouter.Config(
        strategy=LeastLoadedRoutingStrategy.Config()
    ).build(dp_degree=2)
    with pytest.raises(KeyError):
        router.release("missing")


def test_sticky_pins_session_to_dp_rank():
    router = IntraGeneratorRouter.Config(
        strategy=StickySessionRoutingStrategy.Config()
    ).build(dp_degree=3)
    first = router.reserve("r0", routing_session_id="s0")
    # Even though s0's DP rank now has load, the same session sticks to it.
    assert router.reserve("r1", routing_session_id="s0") == first
    # A different session uses the (least-loaded) fallback -> a different DP rank.
    assert router.reserve("r2", routing_session_id="s1") != first


@pytest.mark.parametrize("dp_degree", [0, 1])
def test_rejects_dp_degree_without_routing(dp_degree: int):
    with pytest.raises(ValueError, match="dp_degree must be > 1"):
        IntraGeneratorRouter.Config(
            strategy=RoundRobinRoutingStrategy.Config()
        ).build(dp_degree=dp_degree)
