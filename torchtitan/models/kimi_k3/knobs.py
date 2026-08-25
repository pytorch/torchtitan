# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Topology knobs, resolved once from config.

* These fields decide the pipeline topology (stage count, what crosses a stage
  boundary), so every rank must resolve them identically; a per-rank
  disagreement hangs a collective with nothing pointing at the cause.
* They are read from call sites deep inside the pipeline split where no config
  object is in scope, so the resolved record is module-global:
  ``register_topology`` runs once at the pipelining entry, ``topology()`` reads
  it back.
* First call wins and is idempotent; a second call with a different resolution
  is a real inconsistency and is reported rather than decided by call order.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.tools.logging import logger


@dataclass
class TopologyKnobs:
    """Resolved topology."""

    attn_res_cache: bool = False


_TOPOLOGY: TopologyKnobs | None = None
_WARNED_UNREGISTERED = False


def register_topology(config) -> TopologyKnobs:
    """Resolve the topology from ``config`` once. Idempotent, first call wins.

    A field the config does not carry keeps its default.
    """
    global _TOPOLOGY

    defaults = TopologyKnobs()
    resolved = TopologyKnobs(
        attn_res_cache=bool(
            getattr(config, "attn_res_cache", defaults.attn_res_cache)
        ),
    )
    if _TOPOLOGY is not None and _TOPOLOGY != resolved:
        logger.warning(
            "topology re-registered with a different resolution: keeping %r, "
            "ignoring %r. Two entry points were handed different configs.",
            _TOPOLOGY,
            resolved,
        )
        return _TOPOLOGY
    _TOPOLOGY = resolved
    return _TOPOLOGY


def topology() -> TopologyKnobs:
    """The resolved topology, or the defaults with a warning."""
    global _WARNED_UNREGISTERED

    if _TOPOLOGY is not None:
        return _TOPOLOGY
    if not _WARNED_UNREGISTERED:
        _WARNED_UNREGISTERED = True
        logger.warning(
            "topology knob read before register_topology(); using defaults. "
            "Config fields are NOT being honoured on this path."
        )
    return TopologyKnobs()


def reset_topology_for_testing() -> None:
    """Tests need to re-resolve; production code must not call this."""
    global _TOPOLOGY, _WARNED_UNREGISTERED

    _TOPOLOGY = None
    _WARNED_UNREGISTERED = False
