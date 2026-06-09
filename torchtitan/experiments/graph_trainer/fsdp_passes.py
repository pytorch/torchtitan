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
from torch._inductor.fx_passes.bucketing import (
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_wait_tensor,
)
from torch._inductor.fx_passes.overlap_manual_scheduling import (
    _schedulable_wait_node,
    get_subgraph_by_path,
    make_graph_view,
    manual_overlap_bucketing,
    ManualOverlapPreservingBucketer,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollectiveInfo,
    is_compute_node,
    schedule_overlap_bucketing,
)
from torch._logging import trace_structured
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


class JointManualOverlapScheduler:
    """Manual overlap scheduler for joint forward+backward graphs.

    Lightweight, standalone scheduler that buckets and reorders FSDP
    collectives in a joint forward+backward FX graph.  Unlike the upstream
    ``ManualOverlapScheduler`` (which inherits from ``OverlapScheduler``),
    this class does **not** run ``gather_node_runtime_estimations`` or any
    other expensive/fragile estimation in its constructor.  It only needs
    the graph structure and collective identification — nothing more.

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
        *,
        is_backward_fn: Callable[[fx.Node], bool],
        module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]],
        bucket_mode: BucketMode | None = None,
    ) -> None:
        self.gm = gm
        self.graph = gm.graph
        self.module_bucket_plans = module_bucket_plans
        self.module_stack_fn = module_stack_fn
        self._is_backward_fn = is_backward_fn

        bucket_mode = bucket_mode or "custom_ops"

        self.nodes: list[fx.Node] = list(self.graph.nodes)
        self.node_ancestors: dict[
            fx.Node, OrderedSet[fx.Node]
        ] = self._collect_node_ancestors()

        self.collective_info: dict[fx.Node, CollectiveInfo] = {}
        self._identify_collectives()

        self.in_degree: Counter[fx.Node] = Counter(
            user for node in self.nodes for user in node.users
        )
        self.on_path_ready: list[tuple[int, fx.Node]] = []
        self.scheduled: OrderedSet[fx.Node] = OrderedSet()
        self.nodes_in_subgraph: list[list[fx.Node]] = []

        self.bucketer = ManualOverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=OrderedSet(self.graph.nodes),
            bucket_mode=bucket_mode,
        )

    def _collect_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]
        return ancestors

    def _identify_collectives(self) -> None:
        for node in self.nodes:
            if _schedulable_wait_node(node):
                start = node.args[0]
                self.collective_info[start] = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=0,
                    estimated_time_ms=0,
                    exposed_time_ms=0,
                )

    def _obtain_nodes_in_subgraph(self) -> None:
        graph_view = make_graph_view(self.graph, self.module_stack_fn)
        if graph_view is None:
            return
        for module in self.module_bucket_plans:
            subgraph_view = get_subgraph_by_path(graph_view, module)
            self.nodes_in_subgraph.append(subgraph_view)
        all_subgraph_nodes = [
            node for sublist in self.nodes_in_subgraph for node in sublist
        ]
        unique_subgraph_nodes = list(OrderedSet(all_subgraph_nodes))
        assert len(all_subgraph_nodes) <= len(unique_subgraph_nodes), (
            f"Overlapping FX nodes detected across subgraphs in "
            f"`module_bucket_plans`. Expected disjoint node sets but found "
            f"{len(all_subgraph_nodes) - len(unique_subgraph_nodes)} "
            f"duplicated node(s)."
        )

    def _add_to_ready_queue(self, node: fx.Node) -> None:
        heapq.heappush(self.on_path_ready, (self.node_idx[node], node))

    def _schedule(self, node: fx.Node) -> None:
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                self._add_to_ready_queue(user)

    def run(self) -> fx.GraphModule:
        self._manual_bucket_collectives()
        self._manual_reorder_graph()
        return self.gm

    def _tlparse_dump(self, graph_name: str) -> None:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {"name": graph_name, "encoding": "string"},
            payload_fn=lambda: self.gm.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
                additional_meta=["autograd_backward"],
            ),
            expect_trace_id=False,
        )

    def _manual_bucket_collectives(self) -> None:
        """Bucket per module, splitting by direction to keep fwd/bwd buckets disjoint."""
        self._obtain_nodes_in_subgraph()
        self._tlparse_dump("before_manual_bucket_collectives_pass")
        for nodes in self.nodes_in_subgraph:
            fwd_nodes = [n for n in nodes if not self._is_backward_fn(n)]
            bwd_nodes = [n for n in nodes if self._is_backward_fn(n)]
            if fwd_nodes:
                self.bucketer.manual_bucket_collectives(nodes=fwd_nodes)
            if bwd_nodes:
                self.bucketer.manual_bucket_collectives(nodes=bwd_nodes)
        self._tlparse_dump("after_manual_bucket_collectives_pass")

        self.graph.lint()

        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _manual_reorder_graph(self) -> None:
        """Reorder pass that directly moves nodes instead of relying on
        topological sort. AG start chains are moved earlier (prefetch),
        RS wait chains are moved later (deferred).

        Separate fwd/bwd buffers in ``_schedule_ag_prefetch`` ensure AG
        pairing never crosses the fwd/bwd boundary. RS pairing is
        unchanged — RSs only occur in backward.
        """
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        self._schedule_rs_prefetch(overlap_deps)
        self._schedule_ag_prefetch(overlap_deps)

        self._tlparse_dump("before_reorder_graph_move_pass")
        self._execute_direct_moves(overlap_deps)
        self._tlparse_dump("after_reorder_graph_move_pass")
        self.graph.lint()

    def _execute_direct_moves(
        self, overlap_deps: dict[fx.Node, OrderedSet[fx.Node]]
    ) -> None:
        """Realize overlap_deps by directly moving nodes in the FX graph.

        Classifies each overlap_dep entry by source node type:
        - ``bucketed_reduce_scatter`` sources → RS defer (move wait chain later)
        - ``bucketed_all_gather`` sources → AG prefetch (move start chain earlier)
        """
        rs_defer: dict[fx.Node, list[fx.Node]] = defaultdict(list)
        ag_prefetch: dict[fx.Node, list[fx.Node]] = defaultdict(list)

        for target, sources in overlap_deps.items():
            for source in sources:
                source_type = self.bucketer.bucketed_node_types.get(source, "")
                if source_type == "bucketed_reduce_scatter":
                    rs_defer[target].append(source)
                elif source_type == "bucketed_all_gather":
                    ag_prefetch[target].append(source)

        node_positions = {n: i for i, n in enumerate(self.graph.nodes)}

        # RS defer: move each wait chain to right after the latest RS start
        for rs_wait, rs_starts in rs_defer.items():
            anchor = max(rs_starts, key=lambda n: node_positions[n])
            chain = self._collect_forward_chain(rs_wait)
            logger.info(
                "RS defer: moving %d nodes (%s …) after %s",
                len(chain),
                chain[0].name,
                anchor.name,
            )
            cursor = anchor
            for node in chain:
                cursor.append(node)
                cursor = node

        # AG prefetch: move each start chain to right before the anchor.
        # Skip if the AG start is already before the anchor (e.g. trailing
        # block deps that are no-ops in topological sort).
        for anchor, ag_starts in ag_prefetch.items():
            anchor_pos = node_positions[anchor]
            sorted_starts = sorted(ag_starts, key=lambda n: node_positions[n])
            for ag_start in sorted_starts:
                if node_positions[ag_start] < anchor_pos:
                    continue
                chain = self._collect_ag_chain(ag_start)
                logger.info(
                    "AG prefetch: moving %d nodes (… %s) before %s",
                    len(chain),
                    chain[-1].name,
                    anchor.name,
                )
                for node in chain:
                    anchor.prepend(node)

    @staticmethod
    def _collect_ag_chain(ag_start: fx.Node) -> list[fx.Node]:
        """Collect the AG chain from ``_pre_bucket_all_gather`` through
        ``ag_start`` in topological order.

        The bucketed AG pattern is always::

            _pre_bucket_all_gather → slice → all_gather_into_tensor_out

        Walk forward from the ``_pre_bucket_all_gather`` root, collecting
        every node up to and including ``ag_start``.
        """
        root = ag_start
        while root.op != "placeholder":
            parents = [inp for inp in root.all_input_nodes if inp.op != "placeholder"]
            if not parents:
                break
            root = parents[0]

        chain: list[fx.Node] = [root]
        chain_set: set[fx.Node] = {root}
        i = 0
        while i < len(chain):
            for user in chain[i].users:
                if user not in chain_set and all(
                    inp in chain_set or inp.op == "placeholder"
                    for inp in user.all_input_nodes
                ):
                    chain_set.add(user)
                    chain.append(user)
                    if user is ag_start:
                        return chain
            i += 1
        return chain

    @staticmethod
    def _collect_forward_chain(start: fx.Node) -> list[fx.Node]:
        """Collect ``start`` and all transitively dependent nodes whose
        inputs are all within the chain, in BFS (topological) order.

        For a bucketed RS wait this returns the ``wait_tensor``,
        ``split_with_sizes``, ``getitem`` and ``view`` nodes — everything
        in the unpack chain.
        """
        chain: list[fx.Node] = [start]
        chain_set: set[fx.Node] = {start}
        i = 0
        while i < len(chain):
            for user in chain[i].users:
                if user not in chain_set and all(
                    inp in chain_set for inp in user.all_input_nodes
                ):
                    chain_set.add(user)
                    chain.append(user)
            i += 1
        return chain

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
        :meth:`_schedule_rs_prefetch`.
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
                if is_bwd and node in bwd_scope:
                    bwd_last_compute = node
                elif not is_bwd and node in fwd_scope:
                    fwd_last_compute = node

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
        is_backward_fn=_is_backward_node,
        module_stack_fn=_stack_fn,
        bucket_mode=bucket_mode,
    ).run()
    overlapped_gm.recompile()
    return overlapped_gm
