# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule annotated eager EP exchanges without changing their computation.

1. Source annotations identify each launch and its explicit completion request.
2. Tensor, mutation, and launch-pair dependencies define the legal node order.
3. Within one MoE region, prefer ready launches, then independent work, then
   waits. Work outside that region retains its original order.
4. Apply the planned order with PyTorch's topological sorter and validate it.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._inductor.fx_passes.bucketing import _resolve_group_name

from torchtitan.distributed.deepep.hybridep import (
    _EP_TOKEN_EXCHANGE,
    _EP_TOKEN_EXCHANGE_WAIT,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    ChunkedRegion,
    ChunkOwner,
    collect_chunked_regions,
    ordered_nodes,
)
from torchtitan.experiments.graph_trainer.mutation_utils import (
    base_tensor_for_mutation_target,
    mutation_deps,
)

logger = logging.getLogger(__name__)
_BOUNDARIES = {"placeholder", "get_attr"}


@dataclass(frozen=True)
class _TokenExchange:
    """A launch and the wait consuming its distinct completion request."""

    phase: str
    launch: fx.Node
    wait: fx.Node


def _collect_native_exchanges(
    nodes: Iterable[fx.Node], *, context: str
) -> tuple[_TokenExchange, ...]:
    """Validate source-annotated native exchanges without assuming chunking."""
    allowed_launches = {
        torch.ops.hybridep.dispatch.default: "dispatch",
        torch.ops.hybridep.combine.default: "combine",
        torch.ops.hybridep.dispatch_bwd.default: "dispatch",
        torch.ops.hybridep.combine_bwd.default: "combine",
    }
    launches = {}
    waits = {}
    for node in nodes:
        # Tuple projections inherit traceback annotations, but are not ops.
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        custom = node.meta.get("custom", {})
        phase = custom.get(_EP_TOKEN_EXCHANGE)
        wait_phase = custom.get(_EP_TOKEN_EXCHANGE_WAIT)
        if phase is not None:
            if allowed_launches.get(node.target) != phase:
                raise ValueError(
                    f"Invalid EP launch annotation on {node.name}: {node.target}"
                )
            launches[node] = phase
        if wait_phase is not None:
            if node.target != torch.ops.hybridep.wait.default:
                raise ValueError(
                    f"Invalid EP wait annotation on {node.name}: {node.target}"
                )
            request = node.args[1]
            if not (
                isinstance(request, fx.Node)
                and request.target is operator.getitem
                and isinstance(request.args[0], fx.Node)
            ):
                raise ValueError(
                    f"EP wait {node.name} must consume a launch's completion request"
                )
            launch = request.args[0]
            inputs = {
                input_node
                for input_node in launch.all_input_nodes
                if isinstance(input_node.meta.get("val"), torch.Tensor)
            }
            if set(node.args[2]) != inputs:
                raise ValueError(
                    f"EP wait {node.name} must retain every tensor input of "
                    f"{launch.name}; compiler reuse cannot see opaque references"
                )
            if request.args[1] != len(launch.target._schema.returns) - 1:
                raise ValueError(
                    f"EP wait {node.name} does not consume the completion output"
                )
            if launch in waits:
                raise ValueError(
                    f"EP launch {launch.name} has multiple completion waits"
                )
            waits[launch] = (node, wait_phase)
    if set(launches) != set(waits):
        missing = set(launches) ^ set(waits)
        raise ValueError(
            f"{context} has unmatched EP launches/waits: "
            f"{[node.name for node in missing]}. Enable non-blocking HybridEP async_op."
        )
    exchanges = []
    for launch, phase in launches.items():
        wait, wait_phase = waits[launch]
        if phase != wait_phase:
            raise ValueError(
                f"EP launch/wait phase mismatch: {launch.name} -> {wait.name}"
            )
        exchanges.append(_TokenExchange(phase, launch, wait))
    return tuple(exchanges)


def _collect_exchanges(region: ChunkedRegion) -> dict[int, tuple[_TokenExchange, ...]]:
    """Require one dispatch/combine pair in each annotated eager chunk."""
    exchanges = {}
    for chunk_id, body in region.bodies_by_chunk.items():
        chunk_exchanges = _collect_native_exchanges(
            body.nodes, context=f"EP region {region.root_fqn}, chunk {chunk_id}"
        )
        if len(chunk_exchanges) != 2 or {item.phase for item in chunk_exchanges} != {
            "dispatch",
            "combine",
        }:
            raise ValueError(
                f"EP region {region.root_fqn}, chunk {chunk_id}, requires "
                "one dispatch and one combine with explicit completion waits"
            )
        exchanges[chunk_id] = chunk_exchanges
    signatures = {
        chunk: tuple((exchange.phase, exchange.launch.target) for exchange in values)
        for chunk, values in exchanges.items()
    }
    if signatures[0] != signatures[1]:
        raise ValueError(
            f"EP chunk signatures differ for {region.root_fqn}: {signatures}"
        )
    return exchanges


def _graph_dependencies(gm: fx.GraphModule) -> dict[fx.Node, set[fx.Node]]:
    """Preserve positional arguments, data edges and mutable-storage ordering."""
    deps = {node: set(node.all_input_nodes) for node in gm.graph.nodes}
    # Positional input order is the callable ABI, not a tensor dependency.
    placeholders = gm.graph.find_nodes(op="placeholder")
    for first, second in zip(placeholders, placeholders[1:]):
        deps[second].add(first)
    mutators_by_base = mutation_deps(gm.graph)
    mutable_bases = set(mutators_by_base)
    readers: dict[fx.Node, set[fx.Node]] = {}
    last_write: dict[fx.Node, fx.Node] = {}
    writes: dict[fx.Node, set[fx.Node]] = {}
    for base, mutators in mutators_by_base.items():
        for mutator in mutators:
            writes.setdefault(mutator, set()).add(base)
    for node in gm.graph.nodes:
        bases = {
            base_tensor_for_mutation_target(input_node)
            for input_node in node.all_input_nodes
        } & mutable_bases
        for base in bases:
            if base in last_write:
                deps[node].add(last_write[base])
            readers.setdefault(base, set()).add(node)
        for base in writes.get(node, ()):
            deps[node].update(readers.get(base, set()) - {node})
            readers[base] = set()
            last_write[base] = node
    return deps


def _region_scope(
    region: ChunkedRegion,
    deps: dict[fx.Node, set[fx.Node]],
    owners: dict[fx.Node, ChunkOwner],
) -> set[fx.Node]:
    """Include a region's unowned preparation without pulling in another body."""
    scope = {node for body in region.bodies_by_chunk.values() for node in body.nodes}
    pending = [dep for node in scope for dep in deps[node]]
    while pending:
        node = pending.pop()
        if node in scope or node in owners or node.op in _BOUNDARIES:
            continue
        scope.add(node)
        pending.extend(deps[node])
    return scope


def _prioritized_order(
    deps: dict[fx.Node, set[fx.Node]],
    *,
    order: dict[fx.Node, int],
    launches: set[fx.Node],
    waits: set[fx.Node],
    regions: list[tuple[set[fx.Node], set[fx.Node]]],
    yield_to: dict[fx.Node, set[fx.Node]] | None = None,
) -> list[fx.Node]:
    """Topologically schedule ordered scopes with early launches and late waits.

    Each region contains its eligible scope and the owned body whose completion
    advances to the next region. A multiplexed pair uses one whole-graph scope.
    Optional launch boundaries prefer complementary work among equally ranked
    ready nodes, preventing one source from draining all its completion waits.
    """
    successors: dict[fx.Node, list[fx.Node]] = {node: [] for node in deps}
    for node, predecessors in deps.items():
        for predecessor in predecessors:
            successors[predecessor].append(node)
    remaining = {node: len(predecessors) for node, predecessors in deps.items()}
    ready = {node for node, count in remaining.items() if not count}
    emitted = set()
    result = []
    active_region = 0
    preferred = set()
    while ready:
        while active_region < len(regions) and regions[active_region][1] <= emitted:
            active_region += 1
        scoped_ready = (
            ready & regions[active_region][0] if active_region < len(regions) else set()
        )
        if scoped_ready:
            node = min(
                scoped_ready,
                key=lambda candidate: (
                    0 if candidate in launches else 2 if candidate in waits else 1,
                    candidate not in preferred,
                    order[candidate],
                ),
            )
        else:
            node = min(ready, key=order.__getitem__)
        ready.remove(node)
        emitted.add(node)
        result.append(node)
        if yield_to is not None and node in yield_to:
            preferred = yield_to[node]
        for successor in successors[node]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.add(successor)
    if len(result) != len(deps):
        blocked = [node.name for node in deps if node not in emitted]
        raise ValueError(
            f"Communication scheduling dependencies contain a cycle: {blocked[:8]}"
        )
    return result


def _plan_order(
    gm: fx.GraphModule,
    regions: list[ChunkedRegion],
    exchanges: list[dict[int, tuple[_TokenExchange, ...]]],
) -> list[fx.Node]:
    """Choose the earliest legal launches and postpone waits behind ready work."""
    from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
        _paging_dependencies,
    )

    order = ordered_nodes(gm)
    deps = _graph_dependencies(gm)
    launches, waits = _paging_dependencies(gm, deps)
    for region, chunk_exchanges in zip(regions, exchanges, strict=True):
        chunk_order = (1, 0) if region.is_backward else (0, 1)
        for first, second in zip(
            chunk_exchanges[chunk_order[0]],
            chunk_exchanges[chunk_order[1]],
            strict=True,
        ):
            deps[second.launch].add(first.launch)
            deps[first.wait].add(second.launch)
            launches.update((first.launch, second.launch))
            waits.update((first.wait, second.wait))
    owners = {
        node: body.owner
        for region in regions
        for body in region.bodies_by_chunk.values()
        for node in body.nodes
    }
    scopes = [_region_scope(region, deps, owners) for region in regions]
    body_nodes = [
        {node for body in region.bodies_by_chunk.values() for node in body.nodes}
        for region in regions
    ]
    return _prioritized_order(
        deps,
        order=order,
        launches=launches,
        waits=waits,
        regions=list(zip(scopes, body_nodes, strict=True)),
    )


def _validated_regions(
    gm: fx.GraphModule,
    *,
    module_pattern: str,
) -> tuple[list[ChunkedRegion], list[dict[int, tuple[_TokenExchange, ...]]]]:
    """Collect regions with explicit exchanges and reject incomplete chunk pairs."""
    regions = []
    exchanges = []
    for region in collect_chunked_regions(gm, module_pattern=module_pattern):
        if set(region.bodies_by_chunk) != {0, 1}:
            raise ValueError(
                f"EP region {region.root_fqn} must contain both token chunks"
            )
        pairs = _collect_exchanges(region)
        regions.append(region)
        exchanges.append(pairs)
    if not regions:
        raise ValueError(f"No annotated HybridEP exchanges found for {module_pattern}")
    return regions, exchanges


def ep_overlap_validate_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
) -> fx.GraphModule:
    """Validate real eager launch/request/wait contracts without reordering."""
    del example_inputs
    regions, _ = _validated_regions(
        gm,
        module_pattern=module_pattern,
    )
    from torchtitan.experiments.graph_trainer.fsdp_patterns import fsdp_param_fqns

    roots = {region.root_fqn for region in regions}
    collectives = {
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.all_reduce.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    }
    seen = {}
    for node in gm.graph.nodes:
        if node.target not in collectives:
            continue
        for fqn in fsdp_param_fqns(node):
            if not any(fqn.startswith(root + ".") for root in roots):
                continue
            key = (fqn, node.target, _resolve_group_name(node.args[-1]))
            if key in seen:
                raise ValueError(
                    f"Duplicate eager-chunk FSDP chain for {fqn}: "
                    f"{seen[key].name}, {node.name}. Materialize shared "
                    "parameters outside chunk ownership before tracing."
                )
            seen[key] = node
    return gm


def ep_overlap_schedule_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
) -> fx.GraphModule:
    """Overlap annotated eager EP regions using their explicit dependencies."""
    del example_inputs
    regions, exchanges = _validated_regions(
        gm,
        module_pattern=module_pattern,
    )
    planned = _plan_order(gm, regions, exchanges)
    sequencing = {node: {previous} for previous, node in zip(planned, planned[1:])}
    _stable_topological_sort(gm.graph, sequencing)
    gm.graph.lint()
    gm.recompile()
    if list(gm.graph.nodes) != planned:
        raise ValueError("EP topological sorting did not materialize the planned order")
    logger.info("Scheduled %d annotated eager EP regions", len(regions))
    return gm
