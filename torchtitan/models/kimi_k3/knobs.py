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

    vit_dep: bool = False
    vit_dep_stages: int = 1
    vit_prefetch: int = 0
    # Run the planned encodes in the schedule's idle intervals on the MAIN stream,
    # rather than ahead of time on a side stream. The two are alternatives, not
    # layers: vit_bubble takes over placement when it is on.
    vit_bubble: bool = False
    # One ViT forward in units of one text-stage forward. A parameter and not a
    # runtime measurement, because a plan derived from each rank's own timing
    # would stop being identical across ranks.
    vit_bubble_cost_ratio: float = 0.5
    # How many deferred vision backwards may wait; each holds a micro-batch's
    # tower forward graph alive, so this is the backward half's memory window.
    # 0 is unbounded, the measured-nothing default (see GradQueue).
    vit_bubble_max_pending: int = 0
    vit_tp_heads: bool = True
    attn_res_cache: bool = False


_TOPOLOGY: TopologyKnobs | None = None
_WARNED_UNREGISTERED = False


def register_topology(config) -> TopologyKnobs:
    """Resolve the topology from ``config`` once. Idempotent, first call wins.

    Accepts either the text config or the multimodal one; the multimodal config
    carries the vision knobs itself and reaches the AttnRes cache gate through
    ``kimi_config``. A field the config does not carry keeps its default.
    """
    global _TOPOLOGY

    text_cfg = getattr(config, "kimi_config", config)
    defaults = TopologyKnobs()
    resolved = TopologyKnobs(
        vit_dep=bool(getattr(config, "vit_dep", defaults.vit_dep)),
        vit_dep_stages=int(getattr(config, "vit_dep_stages", defaults.vit_dep_stages)),
        vit_prefetch=int(getattr(config, "vit_prefetch", defaults.vit_prefetch)),
        vit_bubble=bool(getattr(config, "vit_bubble", defaults.vit_bubble)),
        vit_bubble_cost_ratio=float(
            getattr(config, "vit_bubble_cost_ratio", defaults.vit_bubble_cost_ratio)
        ),
        vit_bubble_max_pending=int(
            getattr(config, "vit_bubble_max_pending", defaults.vit_bubble_max_pending)
        ),
        vit_tp_heads=bool(getattr(config, "vit_tp_heads", defaults.vit_tp_heads)),
        attn_res_cache=bool(
            getattr(text_cfg, "attn_res_cache", defaults.attn_res_cache)
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
