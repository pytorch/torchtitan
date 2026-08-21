# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Topology knobs, resolved from config rather than read from the environment.

Finding 32. Five knobs decided the PIPELINE TOPOLOGY from environment variables:
``KIMI_VIT_DEP``, ``KIMI_VIT_DEP_STAGES``, ``KIMI_VIT_PREFETCH``,
``KIMI_VIT_BUBBLE``, ``KIMI_VIT_BUBBLE_COST_RATIO``,
``KIMI_VIT_TP_HEADS`` and ``TORCHTITAN_ATTNRES_CACHE``. Two consequences, and the
first one is the reason this file exists:

* a launcher that exports them non-uniformly gives different ranks different
  topologies, which hangs in a collective with nothing pointing at the cause;
* a run is not reproducible from its config or its checkpoint, and upstream will
  not take env-var topology.

Why a module-level record instead of a parameter
-----------------------------------------------
The three DEP accessors are read from 15 call sites, several of them inside the PP
split where no config is in scope. Threading a config through all of them is the
end state, but a topology is genuinely process-global -- every rank must agree on
it -- so resolving once and reading it back is not the wrong shape, provided the
CONFIG is the source of truth and the resolution point is explicit.

``register_topology`` is therefore called at both entry points that see a config
(``parallelize_kimi_k3`` and ``pipeline_kimi_k3_with_cache_adapter``) and is
idempotent. If a knob is read before any registration, the accessors fall back to
the environment and say so once -- silently reading a default while a config field
said otherwise is the failure mode this file is meant to remove, so it must not be
silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from torchtitan.tools.logging import logger


_WARNED_KNOBS: set[str] = set()
_WARNED_UNREGISTERED = False


def resolve_knob(config, field: str, env: str):
    """A config field, with its retired environment variable still able to override it.

    The field is the source of truth; the env name is honoured because a dozen recorded
    repro commands set it, and silently ignoring them would make every one of those
    documents wrong without saying so. Warned once per variable so the deprecation is
    visible in a log rather than only in a commit message.

    Booleans follow the original convention exactly -- "0" is off, anything else is on --
    so a command that worked before behaves identically.
    """
    default = getattr(config, field)
    raw = os.environ.get(env)
    if raw is None:
        return default
    if env not in _WARNED_KNOBS:
        _WARNED_KNOBS.add(env)
        logger.warning(
            "%s is deprecated; set the config field '%s' instead (this run honours "
            "the environment variable and overrides the config value %r).",
            env,
            field,
            default,
        )
    if isinstance(default, bool):
        return raw != "0"
    return type(default)(raw)


@dataclass
class TopologyKnobs:
    """Resolved topology. Defaults match the historical env-var defaults exactly."""

    vit_dep: bool = False
    vit_dep_stages: int = 1
    vit_prefetch: int = 0
    # Run the planned encodes in the schedule's idle intervals on the MAIN stream,
    # rather than ahead of time on a side stream. The two are alternatives, not
    # layers: vit_bubble takes over placement when it is on.
    vit_bubble: bool = False
    # One ViT forward in units of one text-stage forward, from dep_cost_ratio.py.
    # A parameter and not a runtime measurement, because a plan derived from each
    # rank's own timing would stop being identical across ranks.
    vit_bubble_cost_ratio: float = 0.5
    # How many deferred vision backwards may wait at once. Each one holds a
    # micro-batch's tower forward graph alive, so this is the memory window of the
    # backward half. 0 is unbounded, which is the measured-nothing default -- see
    # GradQueue for why a guessed bound would be worse than none.
    vit_bubble_max_pending: int = 0
    vit_tp_heads: bool = True
    attn_res_cache: bool = False


_TOPOLOGY: TopologyKnobs | None = None


def register_topology(config) -> TopologyKnobs:
    """Resolve the topology from ``config`` once. Idempotent, first call wins.

    Accepts either the text config or the multimodal one; the multimodal config
    carries the vision knobs itself and reaches the AttnRes cache gate through
    ``kimi_config``. First call wins so the two entry points cannot disagree
    depending on which ran first -- a second call with a DIFFERENT resolution is a
    real inconsistency and is reported.
    """
    global _TOPOLOGY

    text_cfg = getattr(config, "kimi_config", config)
    resolved = TopologyKnobs(
        vit_dep=bool(_field(config, "vit_dep", "KIMI_VIT_DEP")),
        vit_dep_stages=int(_field(config, "vit_dep_stages", "KIMI_VIT_DEP_STAGES")),
        vit_prefetch=int(_field(config, "vit_prefetch", "KIMI_VIT_PREFETCH")),
        vit_bubble=bool(_field(config, "vit_bubble", "KIMI_VIT_BUBBLE")),
        vit_bubble_cost_ratio=float(
            _field(config, "vit_bubble_cost_ratio", "KIMI_VIT_BUBBLE_COST_RATIO")
            or 0.5
        ),
        vit_bubble_max_pending=int(
            _field(config, "vit_bubble_max_pending", "KIMI_VIT_BUBBLE_MAX_PENDING")
            or 0
        ),
        vit_tp_heads=bool(_field(config, "vit_tp_heads", "KIMI_VIT_TP_HEADS")),
        attn_res_cache=bool(
            _field(text_cfg, "attn_res_cache", "TORCHTITAN_ATTNRES_CACHE")
        ),
    )
    if _TOPOLOGY is not None and _TOPOLOGY != resolved:
        logger.warning(
            "topology re-registered with a different resolution: keeping %r, ignoring "
            "%r. The two entry points disagree, which means one of them was handed a "
            "different config.",
            _TOPOLOGY,
            resolved,
        )
        return _TOPOLOGY
    _TOPOLOGY = resolved
    return _TOPOLOGY


def _field(config, field: str, env: str):
    """``resolve_knob`` for a config that may not carry the field yet.

    A flavor built before these fields existed still has to run, and for those the
    environment (or the historical default) is all there is.
    """
    if hasattr(config, field):
        return resolve_knob(config, field, env)
    raw = os.environ.get(env)
    default = getattr(TopologyKnobs(), field)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw != "0"
    return type(default)(raw)


def topology() -> TopologyKnobs:
    """The resolved topology, or an environment-derived one with a warning."""
    global _WARNED_UNREGISTERED

    if _TOPOLOGY is not None:
        return _TOPOLOGY
    if not _WARNED_UNREGISTERED:
        _WARNED_UNREGISTERED = True
        logger.warning(
            "topology knob read before register_topology(); falling back to the "
            "environment. Config fields are NOT being honoured on this path."
        )
    return TopologyKnobs(
        vit_dep=os.environ.get("KIMI_VIT_DEP", "0") != "0",
        vit_dep_stages=max(1, int(os.environ.get("KIMI_VIT_DEP_STAGES", "1"))),
        vit_prefetch=max(0, int(os.environ.get("KIMI_VIT_PREFETCH", "0"))),
        vit_bubble=os.environ.get("KIMI_VIT_BUBBLE", "") not in ("", "0"),
        vit_bubble_cost_ratio=float(
            os.environ.get("KIMI_VIT_BUBBLE_COST_RATIO", "0.5")
        ),
        vit_bubble_max_pending=max(
            0, int(os.environ.get("KIMI_VIT_BUBBLE_MAX_PENDING", "0"))
        ),
        vit_tp_heads=os.environ.get("KIMI_VIT_TP_HEADS", "1") != "0",
        attn_res_cache=os.environ.get("TORCHTITAN_ATTNRES_CACHE") == "1",
    )


def reset_topology_for_testing() -> None:
    """Tests need to re-resolve; production code must not call this."""
    global _TOPOLOGY, _WARNED_UNREGISTERED

    _TOPOLOGY = None
    _WARNED_UNREGISTERED = False
    _WARNED_KNOBS.clear()
