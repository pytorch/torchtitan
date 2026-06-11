# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FSDP-specific compiler passes for graph_trainer.

These passes operate on graphs containing SimpleFSDP all-gather/reduce-scatter
collectives.  They are no-ops when the graph contains no FSDP collectives.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._inductor.fx_passes.bucketing import (
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_wait_tensor,
)

try:
    from torch._inductor.fx_passes.overlap_manual_scheduling import _move_overlap_nodes
except ImportError:
    _move_overlap_nodes = None
from torch._inductor.fx_passes.overlap_manual_scheduling import (
    manual_overlap_bucketing,
    ManualOverlapScheduler,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    is_compute_node,
    schedule_overlap_bucketing,
)
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.tools.logging import logger


def is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    """
    Returns True if the node is a wait_tensor node that is the result of an all_gather
    that can be arbitrarily prefetched, i.e., if all its recursive inputs are
    single-input operators that leads to a graph input.
    """
    if is_wait_tensor(node) and is_all_gather(node.args[0]):
        n: torch.fx.Node = node.all_input_nodes[0]
        while len(n.all_input_nodes) == 1:
            if n.all_input_nodes[0].op == "placeholder":
                return True
            n = n.all_input_nodes[0]
    return False


# Maps an FSDP group_name to an extra group_name created by this pass.
# Each NCCL PG gets its own CUDA stream, so the extra PG is what enables
# AG/RS overlap in backward.
_EXTRA_FSDP_PG_REGISTRY: dict[str, str] = {}


def _reorder_overlap_nodes(
    graph: fx.Graph,
    overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    bucketed_node_types: dict[fx.Node, str],
) -> None:
    if _move_overlap_nodes is None:
        # TODO(ivankobzarev): Remove this fallback once the PyTorch nightly wheel
        # includes _move_overlap_nodes.
        _stable_topological_sort(graph, overlap_deps)
    else:
        _move_overlap_nodes(graph, overlap_deps, bucketed_node_types)


def _get_or_create_extra_pg(
    source_pg_name: str,
    registry: dict[str, str],
    *,
    group_desc: str,
    high_priority: bool = False,
) -> str:
    import torch.distributed as dist

    if source_pg_name in registry:
        return registry[source_pg_name]

    source_pg = dist.distributed_c10d._resolve_process_group(source_pg_name)
    ranks = dist.get_process_group_ranks(source_pg)
    pg_options = (
        dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        if high_priority and hasattr(dist, "ProcessGroupNCCL")
        else None
    )
    extra_pg = dist.new_group(
        ranks=ranks,
        backend="nccl" if pg_options is not None else None,
        pg_options=pg_options,
        group_desc=group_desc,
        use_local_synchronization=True,
    )
    registry[source_pg_name] = extra_pg.group_name
    logger.info(
        f"Created extra {group_desc} PG (source: {source_pg_name}, "
        f"extra: {extra_pg.group_name}, high_priority={high_priority})"
    )
    return extra_pg.group_name


def _get_or_create_extra_fsdp_pg(source_pg_name: str) -> str:
    """Return an extra FSDP PG with the same ranks and a distinct NCCL stream."""
    return _get_or_create_extra_pg(
        source_pg_name,
        _EXTRA_FSDP_PG_REGISTRY,
        group_desc="fsdp_extra",
    )


def reassign_collective_pgs_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Reassign collectives to dedicated NCCL process groups.

    Each PG runs on its own CUDA stream, so moving a collective to an extra PG
    (same ranks) lets it overlap with the collectives left on the original PG --
    e.g. all-gathers overlapping reduce-scatters in backward, or isolating EP
    collectives. No-op without targeted collectives; run before bucketing so
    bucketed collectives inherit the new PG.
    """
    source_pg_names: OrderedSet[str] = OrderedSet()
    for node in gm.graph.nodes:
        if is_wait_tensor_from_fsdp(node):
            ag_node = node.args[0]
            source_pg_names.add(ag_node.args[2])

    if not source_pg_names:
        return gm

    pg_mapping: dict[str, str] = {
        pg: _get_or_create_extra_fsdp_pg(pg) for pg in source_pg_names
    }

    ag_count = 0
    for node in gm.graph.nodes:
        if is_all_gather(node) and node.args[2] in pg_mapping:
            # AG args: (input_tensor, group_size, group_name)
            node.args = (node.args[0], node.args[1], pg_mapping[node.args[2]])
            ag_count += 1
    if ag_count > 0:
        for source, target in pg_mapping.items():
            logger.info(f"Rewrote all-gather node(s) from PG {source} to PG {target}")
    gm.recompile()
    return gm


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize comm/compute overlap patterns in the graph.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    fsdp_manual_buckets,
) -> torch.fx.GraphModule:
    """
    Apply aten-level manual bucketing and reordering optimization.
    """
    manual_overlap_bucketing(
        gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False
    )
    gm.recompile()
    return gm


class JointManualOverlapScheduler(ManualOverlapScheduler):
    """Manual overlap scheduler for joint forward+backward graphs.

    For the aot_fx_trace path we trace a joint forward+backward graph and
    want to bucket + reorder both directions in a single pass over the
    graph. This subclass of :class:`ManualOverlapScheduler` produces the
    same bucketing and prefetch pattern as invoking the upstream
    ``manual_overlap_bucketing`` twice (once per direction).

    Overrides :meth:`_manual_bucket_collectives` to split each module's
    collectives by direction before handing them to the bucketer.

    Overrides :meth:`_manual_reorder_graph` to track per-direction state
    so a single reversed walk emits correct AG prefetch edges for both
    forward and backward regions.

    The caller supplies:

    * ``module_stack_fn`` — must return a non-empty module stack for both
      forward and backward nodes belonging to ``module_bucket_plans``
      (i.e. do not filter by direction; that is this class's job).
    * ``is_backward_fn`` — returns ``True`` for nodes that should be
      treated as backward, including SAC-recomputed forward ops that are
      emitted into the backward section.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        module_bucket_plans: list[list[str] | str],
        insert_overlap_deps: bool,
        *,
        is_backward_fn: Callable[[fx.Node], bool],
        module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]],
        bucket_mode: BucketMode | None = None,
    ) -> None:
        super().__init__(
            gm,
            module_bucket_plans,
            insert_overlap_deps,
            module_stack_fn=module_stack_fn,
            bucket_mode=bucket_mode,
        )
        self._is_backward_fn = is_backward_fn

    def _manual_bucket_collectives(self) -> None:
        """Bucket per module, splitting by direction to keep fwd/bwd buckets disjoint."""
        self._obtain_nodes_in_subgraph()
        for nodes in self.nodes_in_subgraph:
            fwd_nodes = [n for n in nodes if not self._is_backward_fn(n)]
            bwd_nodes = [n for n in nodes if self._is_backward_fn(n)]
            if fwd_nodes:
                self.bucketer.manual_bucket_collectives(nodes=fwd_nodes)
            if bwd_nodes:
                self.bucketer.manual_bucket_collectives(nodes=bwd_nodes)

        if _move_overlap_nodes is None:
            _stable_topological_sort(self.graph, {})

        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _manual_reorder_graph(self) -> None:
        """Reorder pass with separate fwd/bwd buffers so AG pairing never
        crosses the fwd/bwd boundary. RS pairing is unchanged — RSs only
        occur in backward and are already direction-scoped.
        """
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        self._schedule_rs_prefetch(overlap_deps)
        self._schedule_ag_prefetch(overlap_deps)

        _reorder_overlap_nodes(
            self.graph, overlap_deps, self.bucketer.bucketed_node_types
        )
        self.graph.lint()

        if self.insert_overlap_deps:
            from torch._inductor.fx_passes.control_dependencies import (
                preserve_node_ordering,
            )

            preserve_node_ordering(self.graph, overlap_deps)

    def _schedule_rs_prefetch(
        self,
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    ) -> None:
        """Top-down scheduling loop that emits RS prefetch edges.

        RSs only occur in backward, so no direction tracking is needed.
        Populates ``self.scheduled`` in topological order for the
        subsequent reversed walk.
        """
        delayed_rs_wait_nodes: list[fx.Node] = []
        current_rs_start_nodes: list[fx.Node] = []

        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.on_path_ready = []
        self.scheduled = OrderedSet()
        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        while self.on_path_ready:
            _, node = heapq.heappop(self.on_path_ready)
            node_type = self.bucketer.bucketed_node_types.get(node, "")

            if node in self.scheduled:
                continue

            if node_type == "bucketed_reduce_scatter":
                current_rs_start_nodes.append(node)
            elif node_type == "bucketed_reduce_scatter_wait":
                if current_rs_start_nodes:
                    for delayed in delayed_rs_wait_nodes:
                        for rs_start in current_rs_start_nodes:
                            overlap_deps[delayed].add(rs_start)
                    delayed_rs_wait_nodes.clear()
                    current_rs_start_nodes.clear()
                delayed_rs_wait_nodes.append(node)

            self._schedule(node)

    def _schedule_ag_prefetch(
        self,
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    ) -> None:
        """Reversed walk that emits per-direction AG prefetch edges.

        Uses separate fwd/bwd buffers so AG pairing never crosses the
        fwd/bwd boundary. Consumes ``self.scheduled`` produced by
        :meth:`_emit_rs_prefetch`.
        """
        self.scheduled = OrderedSet(reversed(list(self.scheduled)))

        bwd_scope: OrderedSet[fx.Node] = OrderedSet()
        fwd_scope: OrderedSet[fx.Node] = OrderedSet()
        for sublist in self.nodes_in_subgraph:
            for n in sublist:
                if self._is_backward_fn(n):
                    bwd_scope.add(n)
                else:
                    fwd_scope.add(n)

        bwd_picked: list[fx.Node] = []
        fwd_picked: list[fx.Node] = []
        bwd_last_compute: fx.Node | None = None
        fwd_last_compute: fx.Node | None = None

        for node in self.scheduled:
            node_type = self.bucketer.bucketed_node_types.get(node, "")
            is_bwd = self._is_backward_fn(node)
            picked = bwd_picked if is_bwd else fwd_picked

            if node_type == "bucketed_all_gather":
                picked.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                if picked:
                    for ag in picked:
                        overlap_deps[self.bucketer.node_to_wait_map[node]].add(ag)
                picked.clear()

            if is_compute_node(node):
                # Track per-direction last_compute so orphan bwd all-gathers
                # attach to a bwd-region compute and orphan fwd all-gathers
                # attach to a fwd-region compute.
                if is_bwd and node in bwd_scope:
                    bwd_last_compute = node
                elif not is_bwd and node in fwd_scope:
                    fwd_last_compute = node

        # Trailing block, applied once per direction. Attaches any orphan
        # AG starts (those whose wait was not matched during the reversed
        # walk of their direction) to the last compute in that direction,
        # unless they are already an ancestor of it.
        self._apply_trailing_block(bwd_picked, bwd_last_compute, overlap_deps)
        self._apply_trailing_block(fwd_picked, fwd_last_compute, overlap_deps)

    def _apply_trailing_block(
        self,
        picked: list[fx.Node],
        last_compute: fx.Node | None,
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    ) -> None:
        if last_compute is None or not picked:
            return
        ancestors = self.node_ancestors
        # TODO(ivankobzarev): remove OrderedSet fallback after nightly picks up BitsetAncestors
        if hasattr(ancestors, "is_ancestor"):
            blocked = any(ancestors.is_ancestor(ag, last_compute) for ag in picked)
        else:
            blocked = bool(OrderedSet(picked) & OrderedSet(ancestors[last_compute]))
        if blocked:
            return
        for ag in picked:
            overlap_deps[last_compute].add(ag)


def joint_transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    module_bucket_plans: list[list[str] | str],
    insert_overlap_deps: bool = False,
    bucket_mode: BucketMode | None = None,
) -> torch.fx.GraphModule:
    """Run joint-graph manual bucketing and reordering.

    Joint-graph equivalent of
    ``torch._inductor.fx_passes.overlap_manual_scheduling.manual_overlap_bucketing``.
    Buckets forward all-gathers, backward all-gathers, and backward reduce-scatters
    of each module into separate buckets per transformer block and emits prefetching.

    Run ``reassign_collective_pgs_pass`` first to put collectives on dedicated
    streams; bucketed collectives inherit the new PGs.

    Args:
        gm: joint forward+backward graph module.
        example_inputs: unused, required by the pass interface.
        module_bucket_plans: list of module FQNs (or lists of FQNs); each
            entry defines one bucketing scope whose collectives should be
            merged into a single bucket per direction per collective type.
        insert_overlap_deps: if ``True``, insert explicit control deps via
            ``preserve_node_ordering`` after the topological sort.
        bucket_mode: bucket mode forwarded to the underlying bucketer;
            defaults to ``"custom_ops"`` via the parent class.
    """

    def _stack_fn(node: torch.fx.Node) -> list[tuple[str, type]]:
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn:
            return []
        return [(fqn, torch.nn.Module)]

    overlapped_gm = JointManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps,
        is_backward_fn=_is_backward_node,
        module_stack_fn=_stack_fn,
        bucket_mode=bucket_mode,
    ).run()
    overlapped_gm.recompile()
    return overlapped_gm
