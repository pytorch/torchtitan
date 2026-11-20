# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routing data types shared by both routing layers in the RL generator and the rollouter.

- ``RoutingCandidate``: a routable target carrying a ``reserved_load``. Each
  caller supplies its own candidate type (``_GeneratorHandle`` for generator
  meshes, ``_DPRankHandle`` for DP ranks, ``_RolloutWorkerLoad`` for the
  rollouter's rollout workers).
- ``RoutingContext``: per-request metadata a strategy may consult.

A ``RoutingStrategy`` (see ``strategies.py``) picks one ``RoutingCandidate``
given a ``RoutingContext``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class RoutingCandidate(ABC):
    """A routable target."""

    @property
    @abstractmethod
    def reserved_load(self) -> int:
        """Caller-defined reserved load on this target."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingContext:
    """Routing metadata for one generation request."""

    session_id: str | None = None
    """Stable session key consumed only by sticky routing strategies; other
    strategies ignore it. ``None`` means the request is unpinned and uses fallback
    routing without session affinity."""
