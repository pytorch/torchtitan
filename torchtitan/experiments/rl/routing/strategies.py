# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routing strategies.

A ``RoutingStrategy`` chooses one ``RoutingCandidate`` from a set given a
``RoutingContext``. It depends only on the ``RoutingCandidate`` base class -- a
``reserved_load`` field plus object identity -- so the same strategy classes
serve both routing layers in the RL generator:

- Layer 1: ``InterGeneratorRouter`` (controller side) routes a call across
  generator *meshes* (replicas). See ``inter_generator_router.py``.
- Layer 2: ``IntraGeneratorRouter`` (in-mesh, rank-0 side) routes a request
  across the *data-parallel ranks* within one generator mesh. See
  ``intra_generator_router.py``.

Each layer supplies its own candidate type (``_GeneratorHandle`` for generators,
``_DPRankHandle`` for DP ranks). The data types live in ``types.py``.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.routing.types import RoutingCandidate, RoutingContext


class RoutingStrategy(Configurable, ABC):
    """Policy object that chooses one candidate for a request.

    Add a new strategy by subclassing this, defining a nested ``Config``, and
    selecting it explicitly in config, e.g.
    ``InterGeneratorRouter.Config(strategy=MyRoutingStrategy.Config())``.
    """

    def __init__(self, config: Configurable.Config):
        # Stateless by default; stateful strategies override __init__.
        del config

    @abstractmethod
    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Choose one candidate from the (non-empty) candidates."""


class RoundRobinRoutingStrategy(RoutingStrategy):
    """Cycle over the candidates in order, ignoring load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config
        self._counter = itertools.count()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the next candidate in round-robin order."""

        del routing_ctx
        return candidates[next(self._counter) % len(candidates)]


class LeastLoadedRoutingStrategy(RoutingStrategy):
    """Pick the candidate with the least reserved load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the candidate with the lowest reserved load."""

        del routing_ctx
        return min(candidates, key=lambda h: h.reserved_load)


class StickySessionRoutingStrategy(RoutingStrategy):
    """Keep requests from the same session on the same candidate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_sessions: int = 4096
        """Maximum number of session-to-candidate assignments to retain,
        evicting least-recently-used sessions first."""

        fallback_strategy: RoutingStrategy.Config = field(
            default_factory=LeastLoadedRoutingStrategy.Config
        )
        """Routing strategy used for new sessions and requests without a session."""

        def __post_init__(self):
            if self.max_sessions <= 0:
                raise ValueError(
                    f"max_sessions must be positive, got {self.max_sessions}"
                )

    def __init__(self, config: Config):
        self._max_sessions = config.max_sessions
        self._fallback_strategy = config.fallback_strategy.build()
        self._sessions: OrderedDict[str, RoutingCandidate] = OrderedDict()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the session's assigned candidate, or assign a new one.

        Unpinned requests (no ``session_id``) and first-seen sessions defer to
        the fallback strategy; a session's first assignment is then remembered so
        every later request with that key reuses the same candidate. If a
        session's pinned candidate is no longer available (e.g. a mesh draining
        for a weight sync), the request falls back and the session is re-pinned to
        the newly chosen candidate. The map is bounded by ``max_sessions`` and
        evicts the least-recently-used session.
        """

        # Unpinned request: no affinity, defer entirely to the fallback.
        if routing_ctx.session_id is None:
            return self._fallback_strategy.choose(routing_ctx, candidates)

        # Reuse the pinned candidate, but only while it is still available
        # (e.g. not draining for a weight sync).
        sticky_candidate = self._sessions.get(routing_ctx.session_id)
        if sticky_candidate is not None:
            if any(h is sticky_candidate for h in candidates):
                # End of the dict means it's the most-recently-used session.
                self._sessions.move_to_end(routing_ctx.session_id)
                return sticky_candidate

        # New session, or the pinned candidate is unavailable: choose via the
        # fallback and (re)pin the session to that candidate.
        chosen = self._fallback_strategy.choose(routing_ctx, candidates)
        self._sessions[routing_ctx.session_id] = chosen
        # End of the dict means it's the most-recently-used session.
        self._sessions.move_to_end(routing_ctx.session_id)
        # Evict the least-recently-used session if the map is full. We assume
        # max_sessions is large enough that active sessions are never the LRU
        # victim (only stale, finished sessions get evicted).
        # TODO: relying solely on max_sessions to avoid premature eviction is
        # easy to implement, but not robust for all scenarios. Revisit with an
        # more robust approach.
        if len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return chosen
