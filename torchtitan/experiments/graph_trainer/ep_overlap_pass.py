# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule already chunked graph_trainer EP regions for communication overlap.

This pass is intentionally separate from graph chunking. ``ep_chunk_pass``
produces a correctness-preserving chunked graph; this file consumes such a graph
and only reorders ready nodes so EP all-to-all launches from paired chunks can
run before the corresponding wait suffixes.

Scheduling contract:
- the graph already contains exactly two chunks for each selected region;
- true chunk body nodes carry ``node.meta["chunk_id"]`` and
  ``node.meta["chunked_region_role"] == "body"``;
- copied body nodes carry ``node.meta["chunked_region_fqn"]`` or enough module
  FQN metadata to recover the selected root;
- MoE dispatcher code annotates EP phases with
  ``node.meta["custom"]["EP"] in {"dispatch", "combine"}``;
- forward chunk bodies are emitted chunk0 then chunk1, while backward bodies are
  emitted chunk1 then chunk0;
- this pass may annotate delayed wait suffix nodes with
  ``node.meta["custom"]["EP_wait"] = True`` and reorder nodes, but it must not
  duplicate nodes, delete nodes, or change tensor values.

A graph chunked by eager tracing or by another producer can reuse this pass if
it satisfies the same metadata and ordering contract. Unsupported or asymmetric
EP regions fail loudly instead of silently producing an invalid schedule.
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch.fx as fx
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    _is_backward_node,
    _ordered_nodes,
)
from torchtitan.tools.logging import logger


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _ep_region(node: fx.Node) -> str | None:
    ep = _custom_meta(node).get("EP")
    return ep if ep in ("dispatch", "combine") else None


def _fqn(node: fx.Node) -> str:
    return _get_module_fqn(node)


def _pattern_root(pattern: str, fqn: str) -> str | None:
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    if len(fqn_parts) < len(pattern_parts):
        return None
    for pattern_part, fqn_part in zip(pattern_parts, fqn_parts):
        if not fnmatch.fnmatchcase(fqn_part, pattern_part):
            return None
    return ".".join(fqn_parts[: len(pattern_parts)])


def _is_all_to_all_node(node: fx.Node) -> bool:
    return node.op == "call_function" and "all_to_all_single" in str(node.target)


@dataclass
class _EpGroup:
    region: str
    nodes: tuple[fx.Node, ...]
    launch: fx.Node


@dataclass(frozen=True)
class _EpCycle:
    phases: tuple[tuple[fx.Node, ...], tuple[fx.Node, ...], tuple[fx.Node, ...]]


def _ep_groups(chunk_nodes: list[fx.Node]) -> list[_EpGroup]:
    groups: list[_EpGroup] = []
    current_region: str | None = None
    current_nodes: list[fx.Node] = []

    def flush() -> None:
        if current_region is None:
            return
        launches = [node for node in current_nodes if _is_all_to_all_node(node)]
        if not launches:
            # EP traceback annotations are source-code scoped. A small
            # bookkeeping island from dispatch/combine can appear before the
            # actual communication group; it is a dependency, not a scheduling
            # anchor. The manual scheduler will pull it in when needed.
            return
        groups.append(
            _EpGroup(
                region=current_region,
                nodes=tuple(current_nodes),
                launch=launches[-1],
            )
        )

    for node in chunk_nodes:
        region = _ep_region(node)
        if region is None:
            continue
        if current_region is None:
            current_region = region
        elif region != current_region:
            flush()
            current_region = region
            current_nodes = []
        current_nodes.append(node)
    flush()
    return groups


def _ep_cycles(
    chunk_nodes: list[fx.Node],
    *,
    order: dict[fx.Node, int],
    mark_wait_suffixes: bool,
) -> list[_EpCycle]:
    """Split one chunk into local dispatch/combine cycles.

    Backward graphs can contain rematerialized forward EP regions followed by
    true backward EP regions. Treating the whole module as one dispatch/combine
    pair misclassifies those nodes, so scheduling works at the local adjacent
    EP-pair level.
    """
    groups: list[_EpGroup] = []
    for group in _ep_groups(chunk_nodes):
        # Dispatch can contain a small split-size all-to-all before the real
        # token exchange. Only the last launch in a same-role run exposes useful
        # overlap; earlier launches are dependencies that get emitted on demand.
        if groups and groups[-1].region == group.region:
            groups[-1] = group
        else:
            groups.append(group)
    if not groups:
        return []
    if len(groups) % 2 != 0:
        raise ValueError(
            "ep_overlap expected paired EP dispatch/combine groups, found "
            f"{[(group.region, group.launch.name) for group in groups]}"
        )

    cycles: list[_EpCycle] = []
    for first, second in zip(groups[0::2], groups[1::2]):
        if first.region == second.region:
            raise ValueError(
                "ep_overlap expected adjacent EP groups to alternate, found "
                f"{first.region} followed by {second.region}"
            )
        first_launch_idx = order[first.launch]
        second_launch_idx = order[second.launch]
        if first_launch_idx >= second_launch_idx:
            raise ValueError(
                "ep_overlap expected local EP launch order to match graph order, "
                f"got {first.launch.name} after {second.launch.name}"
            )

        if mark_wait_suffixes:
            for group in (first, second):
                launch_idx = order[group.launch]
                for node in group.nodes:
                    if order[node] > launch_idx:
                        custom = dict(_custom_meta(node))
                        custom["EP_wait"] = True
                        node.meta["custom"] = custom

        cycles.append(
            _EpCycle(
                phases=(
                    tuple(node for node in first.nodes if order[node] <= first_launch_idx),
                    tuple(
                        [node for node in first.nodes if order[node] > first_launch_idx]
                        + [
                            node
                            for node in second.nodes
                            if order[node] <= second_launch_idx
                        ]
                    ),
                    tuple(
                        node
                        for node in second.nodes
                        if order[node] > second_launch_idx
                    ),
                )
            )
        )
    return cycles


def _observed_ep_phase_chunk_order(
    phases_by_chunk: dict[int, _EpCycle],
    order: dict[fx.Node, int],
) -> tuple[int, int]:
    return tuple(
        sorted(
            phases_by_chunk,
            key=lambda chunk_id: min(
                order[node] for node in phases_by_chunk[chunk_id].phases[0]
            ),
        )
    )


@dataclass(frozen=True)
class _ScheduledRegion:
    root: str
    is_backward: bool
    phases: tuple[tuple[fx.Node, ...], ...]


def _move_nodes_in_order(graph: fx.Graph, nodes: list[fx.Node]) -> None:
    cursor = None
    for node in nodes:
        if cursor is not None and cursor.next is not node:
            cursor.append(node)
        cursor = node


def _manual_block_schedule(
    graph: fx.Graph,
    scheduled_regions: list[_ScheduledRegion],
    order: dict[fx.Node, int],
) -> None:
    """Emit explicit EP blocks and pull in ordinary dependencies as needed.

    A priority topological sort can legally leave the graph unchanged when a
    chunk body depends on setup nodes that do not themselves carry chunk-body
    metadata. Eager chunking has exactly that shape. This scheduler instead
    emits the requested phase blocks directly. While emitting a block it may
    recursively emit non-body dependencies, but it refuses to pull a future
    body phase across the requested schedule.
    """
    owner: dict[fx.Node, tuple[_ScheduledRegion, int]] = {}
    anchors: dict[fx.Node, _ScheduledRegion] = {}
    for region in scheduled_regions:
        body_nodes = [node for phase in region.phases for node in phase]
        anchors[min(body_nodes, key=order.__getitem__)] = region
        for phase_idx, phase in enumerate(region.phases):
            for node in phase:
                owner[node] = (region, phase_idx)

    emitted: set[fx.Node] = set()
    new_order: list[fx.Node] = []

    def emit(
        node: fx.Node,
        *,
        active_region: _ScheduledRegion | None = None,
        active_phase: int | None = None,
    ) -> None:
        if node in emitted:
            return
        if node in owner:
            region, phase_idx = owner[node]
            if (
                active_region is not region
                or active_phase is None
                or phase_idx > active_phase
            ):
                direction = "backward" if region.is_backward else "forward"
                raise ValueError(
                    "ep_overlap cannot emit requested schedule because "
                    f"{node.name} from phase {phase_idx} of {region.root!r} "
                    f"({direction}) is needed before that phase."
                )

        for inp in sorted(node.all_input_nodes, key=order.__getitem__):
            emit(inp, active_region=active_region, active_phase=active_phase)
        emitted.add(node)
        new_order.append(node)

    for node in list(graph.nodes):
        if node in emitted:
            continue
        region = anchors.get(node)
        if region is not None:
            for phase_idx, phase in enumerate(region.phases):
                for phase_node in sorted(phase, key=order.__getitem__):
                    emit(
                        phase_node,
                        active_region=region,
                        active_phase=phase_idx,
                    )
        elif node in owner:
            continue
        else:
            emit(node)

    if len(new_order) != len(list(graph.nodes)):
        raise AssertionError("ep_overlap manual block schedule dropped graph nodes")
    _move_nodes_in_order(graph, new_order)


def _validate_region_phase_order(
    scheduled_regions: list[_ScheduledRegion],
    order: dict[fx.Node, int],
) -> None:
    for region in scheduled_regions:
        previous_max: int | None = None
        for phase_idx, phase in enumerate(region.phases):
            if not phase:
                continue
            phase_min = min(order[node] for node in phase)
            phase_max = max(order[node] for node in phase)
            if previous_max is not None and previous_max >= phase_min:
                direction = "backward" if region.is_backward else "forward"
                raise ValueError(
                    "ep_overlap failed to materialize requested block order for "
                    f"{region.root!r} ({direction}) at phase {phase_idx}."
                )
            previous_max = phase_max


def _validate_remat_not_advanced(
    old_order: dict[fx.Node, int],
    new_order: dict[fx.Node, int],
) -> None:
    remat_policies = (
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_CPU_OFFLOAD,
    )
    advanced = [
        node
        for node in old_order
        if node.meta.get("autograd_backward") is True
        and node.meta.get("recompute") in remat_policies
        and new_order[node] < old_order[node]
    ]
    if advanced:
        examples = ", ".join(node.name for node in advanced[:5])
        raise ValueError(
            "ep_overlap must not move SAC rematerialization earlier than the "
            f"remat pass placed it; advanced nodes: {examples}"
        )


def _schedule_ep_overlap_regions(
    gm: fx.GraphModule,
    *,
    module_pattern: str,
    require_all_to_all: bool,
    reorder: bool = True,
) -> int:
    """Order chunked regions so EP all-to-all launches precede peer waits.

    Each chunk is split into three scheduling phases. The boundary is the last
    all-to-all in each EP phase; for forward dispatch this intentionally means
    the token-exchange all-to-all, not the earlier split-size exchange.
    1. compute through the first overlap-relevant EP all-to-all launch,
    2. first wait suffix, expert compute, and second overlap-relevant launch,
    3. second wait suffix and post-EP tail.

    Forward regions use dispatch then combine. Backward autograd sees combine
    gradients before dispatch gradients, so it uses combine then dispatch and
    reverses the chunk order. Non-MoE transformer blocks have no EP phases and
    are left in the chunk pass's original topological order.
    """
    order = _ordered_nodes(gm)
    grouped: dict[tuple[str, bool], dict[int, list[fx.Node]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for node in gm.graph.nodes:
        chunk_id = node.meta.get("chunk_id")
        if chunk_id not in (0, 1):
            continue
        if node.meta.get("chunked_region_role") != "body":
            continue
        root = node.meta.get("chunked_region_fqn")
        if not isinstance(root, str) or not root:
            fqn = _fqn(node)
            root = _pattern_root(module_pattern, fqn) or ""
        if not root:
            continue
        grouped[(root, _is_backward_node(node))][chunk_id].append(node)

    scheduled_regions: list[_ScheduledRegion] = []
    scheduled = 0
    for (root, is_backward), by_chunk in sorted(
        grouped.items(),
        key=lambda item: min(order[n] for nodes in item[1].values() for n in nodes),
    ):
        if set(by_chunk) != {0, 1}:
            raise ValueError(
                f"ep_overlap expected both chunk 0 and chunk 1 for region {root!r} "
                f"({'backward' if is_backward else 'forward'}), found {sorted(by_chunk)}"
            )

        cycles_by_chunk = {
            chunk_id: _ep_cycles(
                sorted(by_chunk[chunk_id], key=order.__getitem__),
                order=order,
                mark_wait_suffixes=reorder,
            )
            for chunk_id in (0, 1)
        }
        missing_cycles = [
            chunk_id for chunk_id, cycles in cycles_by_chunk.items() if not cycles
        ]
        if missing_cycles:
            if len(missing_cycles) == len(cycles_by_chunk):
                continue
            raise ValueError(
                f"ep_overlap found EP all-to-all regions for only one chunk of "
                f"{root!r} ({'backward' if is_backward else 'forward'}): "
                f"missing chunks {missing_cycles}."
            )
        if len(cycles_by_chunk[0]) != len(cycles_by_chunk[1]):
            raise ValueError(
                f"ep_overlap expected the same number of EP cycles in both "
                f"chunks for {root!r} ({'backward' if is_backward else 'forward'}), "
                f"found chunk0={len(cycles_by_chunk[0])} "
                f"chunk1={len(cycles_by_chunk[1])}."
            )

        for cycle_idx in range(len(cycles_by_chunk[0])):
            cycle_by_chunk = {
                chunk_id: cycles[cycle_idx]
                for chunk_id, cycles in cycles_by_chunk.items()
            }
            chunk_order = _observed_ep_phase_chunk_order(cycle_by_chunk, order)
            first = cycle_by_chunk[chunk_order[0]].phases
            second = cycle_by_chunk[chunk_order[1]].phases
            ordered_phases = [
                tuple(first[0]),
                tuple(second[0]),
                tuple(first[1]),
                tuple(second[1]),
                tuple(first[2]),
                tuple(second[2]),
            ]
            scheduled_regions.append(
                _ScheduledRegion(
                    root=root,
                    is_backward=is_backward,
                    phases=tuple(ordered_phases),
                )
            )
            scheduled += 1

    if scheduled and reorder:
        _manual_block_schedule(gm.graph, scheduled_regions, order)
        gm.graph.lint()
        gm.recompile()
        new_order = _ordered_nodes(gm)
        _validate_region_phase_order(scheduled_regions, new_order)
        _validate_remat_not_advanced(order, new_order)
    elif require_all_to_all:
        raise ValueError(
            f"ep_overlap did not find any chunked EP all-to-all regions for "
            f"pattern {module_pattern}."
        )
    return scheduled


def ep_overlap_validate_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = False,
) -> fx.GraphModule:
    """Validate the already chunked graph without changing node order."""
    del example_inputs
    validated = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
        reorder=False,
    )
    logger.info(
        "Validated %d ep_overlap chunked region(s): module=%s",
        validated,
        module_pattern,
    )
    return gm


def ep_overlap_schedule_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = True,
) -> fx.GraphModule:
    """Reorder already chunked regions around EP all-to-alls."""
    del example_inputs
    scheduled = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
    )
    logger.info(
        "Applied ep_overlap scheduling to %d chunked region(s): module=%s",
        scheduled,
        module_pattern,
    )
    return gm
