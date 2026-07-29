# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Named custom schedules for chunked EP-overlap regions.

A provider receives the configured memory policy and the exchange/module
structure observed in one two-chunk region. It returns anchors in the requested
execution order: a token anchor selects one exact launch, while a module-FQN
anchor selects that module subtree in one chunk. A ready-filler anchor emits
currently ready work not claimed by another anchor, without crossing a token
exchange wait. Returning ``None`` declares the region unsupported and permits
the scheduler to fall back to ``auto``; returning incomplete or
dependency-invalid anchors is a policy error.
"""

from __future__ import annotations

import fnmatch

from dataclasses import dataclass
from typing import Literal, TypeAlias

from torchtitan.experiments.graph_trainer.registry import (
    EP_OVERLAP_SCHEDULE_REGISTRY,
    register_ep_overlap_schedule,
)


EpOverlapExecution = Literal["forward", "recompute", "backward"]
EpTokenExchangePhase = Literal["dispatch", "combine"]


@dataclass(frozen=True, slots=True)
class TokenExchangeAnchor:
    execution: EpOverlapExecution
    phase: EpTokenExchangePhase
    chunk_id: Literal[0, 1]
    occurrence: int = 0


@dataclass(frozen=True, slots=True)
class ModuleFQNAnchor:
    execution: EpOverlapExecution
    module_fqn: str
    chunk_id: Literal[0, 1]


@dataclass(frozen=True, slots=True)
class ReadyFillerAnchor:
    execution: EpOverlapExecution


EpOverlapScheduleAnchor: TypeAlias = (
    TokenExchangeAnchor | ModuleFQNAnchor | ReadyFillerAnchor
)
EpExchangeSignature: TypeAlias = tuple[
    tuple[EpOverlapExecution, EpTokenExchangePhase], ...
]


@dataclass(frozen=True, slots=True)
class CustomScheduleContext:
    root_fqn: str
    direction: Literal["forward", "backward"]
    memory_policy: str
    exchange_signature: EpExchangeSignature
    module_fqns: frozenset[str]


def matches_module_fqn_subtree(pattern: str, fqn: str) -> bool:
    """Match a module-FQN pattern against a module and its descendants."""
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    return len(fqn_parts) >= len(pattern_parts) and all(
        fnmatch.fnmatchcase(fqn_part, pattern_part)
        for pattern_part, fqn_part in zip(pattern_parts, fqn_parts)
    )


def validate_ep_overlap_schedule_name(name: str) -> None:
    if name != "auto" and name not in EP_OVERLAP_SCHEDULE_REGISTRY:
        available = ["auto", *EP_OVERLAP_SCHEDULE_REGISTRY]
        raise ValueError(
            f"Unknown EP-overlap schedule {name!r}. Available: {available}."
        )


_FORWARD_SIGNATURE: EpExchangeSignature = (
    ("forward", "dispatch"),
    ("forward", "combine"),
)
_BACKWARD_SIGNATURE: EpExchangeSignature = (
    ("backward", "dispatch"),
    ("backward", "combine"),
)
_FULL_RECOMPUTE_SIGNATURE: EpExchangeSignature = (
    ("recompute", "dispatch"),
    ("recompute", "combine"),
    *_BACKWARD_SIGNATURE,
)


def _token(
    execution: EpOverlapExecution,
    phase: EpTokenExchangePhase,
    chunk_id: Literal[0, 1],
) -> TokenExchangeAnchor:
    return TokenExchangeAnchor(execution, phase, chunk_id)


def _auto_phase(
    execution: EpOverlapExecution,
    *,
    chunk_order: tuple[Literal[0, 1], Literal[0, 1]],
) -> tuple[EpOverlapScheduleAnchor, ...]:
    first, second = chunk_order
    return (
        _token(execution, "dispatch", first),
        _token(execution, "dispatch", second),
        ReadyFillerAnchor(execution),
        _token(execution, "combine", first),
        _token(execution, "combine", second),
        ReadyFillerAnchor(execution),
    )


@register_ep_overlap_schedule("deepseek_v3")
def deepseek_v3_schedule(
    context: CustomScheduleContext,
) -> tuple[EpOverlapScheduleAnchor, ...] | None:
    """Express the greedy auto order with explicit token and filler anchors."""
    routed_experts_fqn = f"{context.root_fqn}.routed_experts.inner_experts"
    if not any(
        matches_module_fqn_subtree(routed_experts_fqn, fqn)
        for fqn in context.module_fqns
    ):
        return None
    if context.direction == "forward":
        if context.exchange_signature != _FORWARD_SIGNATURE:
            return None
        return _auto_phase("forward", chunk_order=(0, 1))

    if context.exchange_signature == _BACKWARD_SIGNATURE:
        return _auto_phase("backward", chunk_order=(1, 0))
    if context.exchange_signature == _FULL_RECOMPUTE_SIGNATURE:
        return (
            _token("recompute", "dispatch", 1),
            _token("recompute", "dispatch", 0),
            ReadyFillerAnchor("backward"),
            ReadyFillerAnchor("recompute"),
            ReadyFillerAnchor("backward"),
            _token("recompute", "combine", 1),
            _token("recompute", "combine", 0),
        ) + _auto_phase("backward", chunk_order=(1, 0))
    return None
