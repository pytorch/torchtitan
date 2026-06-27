# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Intra-generator (in-mesh) DP request routing."""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.routing.strategies import (
    RoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.experiments.rl.routing.types import RoutingCandidate, RoutingContext


@dataclass(kw_only=True, slots=True)
class _DPRankHandle(RoutingCandidate):
    """One data-parallel rank as a routing candidate (satisfies ``RoutingCandidate``)."""

    dp_rank: int
    """vLLM data_parallel_rank."""

    reserved_load: int = 0
    """Number of in-flight requests currently routed to this DP rank."""


class IntraGeneratorRouter(Configurable):
    """Router that partitions requests across the DP ranks within one generator.

    This is layer 2 of the two-layer routing design: the controller-side
    ``InterGeneratorRouter`` routes a call across generators; this router
    then routes each request across the DP ranks within one generator.

    Load is measured as in-flight request count: each ``reserve`` adds one unit
    of load to the chosen DP rank, and the matching ``release`` removes it. The
    request_id -> DP rank map is kept internal, so callers reserve and release a
    request purely by its ``request_id`` and never track the chosen rank or its
    load themselves.

    This class is not thread-safe. Drive a router instance from one event loop
    or otherwise synchronize access to ``reserve`` and ``release``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        strategy: RoutingStrategy.Config = field(
            default_factory=StickySessionRoutingStrategy.Config
        )
        """In-mesh DP routing strategy: how rank 0 partitions requests across the
        generator's data-parallel ranks. Selected by its config type, e.g.
        ``StickySessionRoutingStrategy.Config()`` (default, for prefix-cache
        reuse) or ``LeastLoadedRoutingStrategy.Config()``."""

    def __init__(self, config: Config, *, dp_degree: int):
        if dp_degree <= 1:
            raise ValueError(f"dp_degree must be > 1, got {dp_degree}")
        self._strategy = config.strategy.build()
        self._handles = [_DPRankHandle(dp_rank=rank) for rank in range(dp_degree)]
        # request_id -> the DP rank reserved for that request, so ``release`` can
        # free the load on the same rank when the request's completion resolves.
        self._reservations: dict[str, int] = {}

    def reserve(self, request_id: str, *, routing_session_id: str | None) -> int:
        """Pick a DP rank for one request and reserve one load unit on it.

        Records the choice under ``request_id`` so a later ``release`` frees the
        same rank. Returns the chosen vLLM DP rank.
        """

        assert (
            request_id not in self._reservations
        ), f"request_id {request_id!r} already has a reservation"

        ctx = RoutingContext(session_id=routing_session_id)
        handle = self._strategy.choose(ctx, self._handles)
        handle.reserved_load += 1
        self._reservations[request_id] = handle.dp_rank
        return handle.dp_rank

    def release(self, request_id: str) -> None:
        """Free the load unit ``reserve`` placed for ``request_id``."""

        dp_rank = self._reservations.pop(request_id)
        handle = self._handles[dp_rank]
        handle.reserved_load -= 1
        assert (
            handle.reserved_load >= 0
        ), f"dp rank {dp_rank} reserved_load went negative: {handle.reserved_load}"
