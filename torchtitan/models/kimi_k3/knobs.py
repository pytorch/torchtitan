# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Topology knobs, resolved from config rather than read from the environment.

Finding 32. The cross-stage cache is gated by ``TORCHTITAN_ATTNRES_CACHE``, an
environment variable. Two consequences, and the first one is the reason this
file exists:

* a launcher that exports it non-uniformly gives different ranks different
  topologies, which hangs in a collective with nothing pointing at the cause;
* a run is not reproducible from its config or its checkpoint, and upstream will
  not take env-var topology.

``register_topology`` is called at the entry point that sees a config
(``pipeline_kimi_k3_with_cache_adapter``) and is idempotent. If a knob is read
before any registration, the accessor falls back to the environment and says so
once -- silently reading a default while a config field said otherwise is the
failure mode this file is meant to remove, so it must not be silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from torchtitan.tools.logging import logger


_WARNED_KNOBS: set[str] = set()
_WARNED_UNREGISTERED = False


def resolve_knob(config, field: str, env: str):
    """A config field, with its retired environment variable still able to override it.

    The field is the source of truth; the env name is honoured because recorded
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

    attn_res_cache: bool = False


_TOPOLOGY: TopologyKnobs | None = None


def register_topology(config) -> TopologyKnobs:
    """Resolve the topology from ``config`` once. Idempotent, first call wins.

    Accepts either the text config or a wrapper carrying it as ``kimi_config``.
    First call wins so repeated entry cannot disagree depending on which ran
    first -- a second call with a DIFFERENT resolution is a real inconsistency
    and is reported.
    """
    global _TOPOLOGY

    text_cfg = getattr(config, "kimi_config", config)
    resolved = TopologyKnobs(
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

    A flavor built before this field existed still has to run, and for those the
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
        attn_res_cache=os.environ.get("TORCHTITAN_ATTNRES_CACHE") == "1",
    )


def reset_topology_for_testing() -> None:
    """Tests need to re-resolve; production code must not call this."""
    global _TOPOLOGY, _WARNED_UNREGISTERED

    _TOPOLOGY = None
    _WARNED_UNREGISTERED = False
    _WARNED_KNOBS.clear()
