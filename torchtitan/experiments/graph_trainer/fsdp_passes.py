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
import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
    ManualOverlapPreservingBucketer,
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


_FSDP_BUCKET_META = "fsdp_bucket"


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


def get_fsdp_param_module_order(state_fqns: list[str]) -> dict[str, int]:
    """Return module order matching FSDP2's first-seen parameter order."""
    order: dict[str, int] = {}
    for fqn in state_fqns:
        if "." not in fqn:
            continue
        module_fqn = fqn.rsplit(".", 1)[0]
        order.setdefault(module_fqn, len(order))
    return order


class FSDPParamOrderBucketer(ManualOverlapPreservingBucketer):
    """Pack FSDP buckets in Eager FSDP2 parameter order."""

    def __init__(
        self,
        *args: Any,
        fsdp_param_module_order: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fsdp_param_module_order = fsdp_param_module_order or {}

    def _param_order_key(self, node: fx.Node) -> tuple[int, int]:
        module_fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        param_idx = self.fsdp_param_module_order.get(module_fqn, len(self.node_idx))
        return (param_idx, self.node_idx[node])

    def _bucket_group(self, coll_nodes: list[fx.Node]) -> None:
        if self.fsdp_param_module_order:
            coll_nodes = sorted(coll_nodes, key=self._param_order_key)
        return super()._bucket_group(coll_nodes)


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
        fsdp_param_module_order: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            gm,
            module_bucket_plans,
            insert_overlap_deps,
            module_stack_fn=module_stack_fn,
            bucket_mode=bucket_mode,
        )
        self._is_backward_fn = is_backward_fn
        effective_bucket_mode = self.bucketer.bucket_mode
        self.bucketer = FSDPParamOrderBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=OrderedSet(self.graph.nodes),
            bucket_mode=effective_bucket_mode,
            fsdp_param_module_order=fsdp_param_module_order,
        )

    def _manual_bucket_collectives(self) -> None:
        """Bucket per module, splitting by direction to keep fwd/bwd buckets disjoint."""
        self._obtain_nodes_in_subgraph()
        for bucket_plan, nodes in zip(
            self.module_bucket_plans, self.nodes_in_subgraph, strict=True
        ):
            bucket_fqns = _bucket_plan_fqns(bucket_plan)
            fwd_nodes = [n for n in nodes if not self._is_backward_fn(n)]
            bwd_nodes = [n for n in nodes if self._is_backward_fn(n)]
            if fwd_nodes:
                pre_nodes = set(self.graph.nodes)
                self.bucketer.manual_bucket_collectives(nodes=fwd_nodes)
                self._annotate_new_bucket_nodes(pre_nodes, bucket_fqns, "fwd")
            if bwd_nodes:
                pre_nodes = set(self.graph.nodes)
                self.bucketer.manual_bucket_collectives(nodes=bwd_nodes)
                self._annotate_new_bucket_nodes(pre_nodes, bucket_fqns, "bwd")

        if _move_overlap_nodes is None:
            _stable_topological_sort(self.graph, {})

        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _annotate_new_bucket_nodes(
        self,
        pre_nodes: set[fx.Node],
        bucket_fqns: tuple[str, ...],
        direction: str,
    ) -> None:
        # The upstream bucketer preserves sample ``custom`` metadata for
        # readability, but bucket ownership is the bucketing plan itself. Store
        # that provenance once, at bucket creation time, so later scheduling
        # passes do not need to rediscover it by walking arbitrary users.
        for node in self.graph.nodes:
            if node in pre_nodes or node.op != "call_function":
                continue
            node.meta[_FSDP_BUCKET_META] = {
                "plan_fqns": bucket_fqns,
                "direction": direction,
            }
            if direction == "bwd":
                node.meta["autograd_backward"] = True

    def _manual_reorder_graph(self) -> None:
        """Reorder pass with separate fwd/bwd buffers so AG pairing never
        crosses the fwd/bwd boundary. RS pairing is unchanged — RSs only
        occur in backward and are already direction-scoped.
        """
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        self._schedule_rs_prefetch(overlap_deps)
        self._schedule_ag_prefetch(overlap_deps)

        total_deps = sum(len(v) for v in overlap_deps.values())
        logger.info(
            "FSDP reorder: %d overlap deps across %d target nodes",
            total_deps,
            len(overlap_deps),
        )
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

            if node_type in ("bucketed_reduce_scatter", "bucketed_all_reduce"):
                current_rs_start_nodes.append(node)
            elif node_type in (
                "bucketed_reduce_scatter_wait",
                "bucketed_all_reduce_wait",
            ):
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
    fsdp_param_module_order: dict[str, int] | None = None,
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
        fsdp_param_module_order: module order derived from traced parameter
            FQNs, used to pack FSDP buckets like Eager FSDP2.
    """

    def _stack_fn(node: torch.fx.Node) -> list[tuple[str, type]]:
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn:
            return []
        return [(fqn, torch.nn.Module)]

    scheduler = JointManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps,
        is_backward_fn=_is_backward_node,
        module_stack_fn=_stack_fn,
        bucket_mode=bucket_mode,
        fsdp_param_module_order=fsdp_param_module_order,
    )
    overlapped_gm = scheduler.run()
    overlapped_gm.recompile()
    return overlapped_gm


# --- EP-aware FSDP dense-region scheduling ---


@dataclass(frozen=True)
class _FsdpComm:
    launch: fx.Node
    wait: fx.Node
    kind: str
    direction: str
    plan_fqns: tuple[str, ...]
    logical_index: int | None


def _bucket_plan_fqns(bucket: list[str] | str) -> tuple[str, ...]:
    return (bucket,) if isinstance(bucket, str) else tuple(bucket)


def _layer_id_from_fqn(fqn: str) -> int | None:
    """Extract layer index from ``layers.N...`` FQN prefix."""
    if not fqn.startswith("layers."):
        return None
    parts = fqn.split(".", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def get_transformer_block_bucket_counts(
    module_bucket_plans: list[list[str] | str],
    *,
    n_layers: int,
) -> dict[int, int]:
    """Count how many bucket-plan entries belong to each transformer block."""
    bucket_counts = {layer_id: 0 for layer_id in range(n_layers)}
    for bucket in module_bucket_plans:
        fqns = [bucket] if isinstance(bucket, str) else bucket
        layer_ids = {
            layer_id
            for fqn in fqns
            if (layer_id := _layer_id_from_fqn(fqn)) is not None
        }
        if not layer_ids:
            continue
        if len(layer_ids) != 1:
            raise ValueError(
                "Transformer block bucket plans must not span multiple layers, "
                f"got bucket {bucket!r} with layer ids {sorted(layer_ids)}."
            )
        layer_id = next(iter(layer_ids))
        if layer_id not in bucket_counts:
            raise ValueError(
                f"Transformer block bucket plan references layer {layer_id}, "
                f"but n_layers={n_layers}."
            )
        bucket_counts[layer_id] += 1
    missing = [layer_id for layer_id, count in bucket_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "Transformer block bucket plan is missing layers "
            f"{missing} for n_layers={n_layers}."
        )
    return bucket_counts


def _is_moe_layer_dense_fqn(fqn: str) -> bool:
    """Return whether an MoE-layer FQN belongs to the dense attention region."""
    parts = fqn.split(".")
    if len(parts) < 3:
        return False
    return parts[2] in {"attention", "attention_norm"}


def _is_top_level_param_fqn(fqn: str) -> bool:
    return fqn in {"norm", "lm_head"} or fqn.startswith(("norm.", "lm_head."))


def _fsdp_bucket_logical_index(
    plan_fqns: tuple[str, ...],
    *,
    n_layers: int,
) -> int | None:
    """Map bucket provenance to the logical model slot it owns.

    ``0..N-1`` are transformer blocks. ``N`` is the post-transformer
    norm/lm_head bucket. Embedding buckets return ``None`` because the dense
    scheduler intentionally leaves the bottom edge to the upstream bucketer.
    """
    layer_ids = {
        layer_id
        for fqn in plan_fqns
        if (layer_id := _layer_id_from_fqn(fqn)) is not None
    }
    if len(layer_ids) > 1:
        raise ValueError(
            "FSDP dense-region scheduling cannot place a bucket spanning "
            f"multiple transformer blocks: {plan_fqns!r}"
        )
    if layer_ids:
        layer_id = next(iter(layer_ids))
        if layer_id >= n_layers:
            raise ValueError(
                f"FSDP bucket references layer {layer_id}, but n_layers={n_layers}."
            )
        return layer_id
    if any(_is_top_level_param_fqn(fqn) for fqn in plan_fqns):
        return n_layers
    return None


def _is_dense_region_target_node(node: fx.Node) -> bool:
    """Return whether ``node`` is useful dense compute, not FSDP plumbing.

    Dense-region scheduling uses these nodes as insertion anchors.  FSDP waits
    and parameter-unpack view/copy nodes are part of getting parameters ready;
    choosing them as anchors hoists future AG/RS launches out of the actual
    compute window and can make them interfere with MoE token exchange.
    """
    if node.op != "call_function":
        return False
    if is_wait_tensor(node) or is_all_gather(node):
        return False
    if node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default:
        return False
    namespace = getattr(node.target, "namespace", None)
    if namespace in {"_c10d_functional", "c10d_functional", "c10d"}:
        return False
    if namespace == "bucketing":
        return False

    non_compute_targets = {
        operator.getitem,
        torch.ops.aten._to_copy.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.view.default,
        torch.ops.aten.view.dtype,
    }
    if node.target in non_compute_targets:
        return False

    dense_compute_targets = {
        torch.ops.aten._fused_rms_norm.default,
        torch.ops.aten._fused_rms_norm_backward.default,
        torch.ops.aten.relu.default,
    }
    return is_compute_node(node) or node.target in dense_compute_targets


def _is_backward_or_recomputed_node(node: fx.Node) -> bool:
    """Return whether ``node`` executes in the backward section."""
    return _is_backward_node(node) or "_recomputed" in node.name


def _build_layer_dense_regions(
    gm: torch.fx.GraphModule,
    n_layers: int,
    moe_layer_ids: frozenset[int],
) -> dict[int, dict[str, list[fx.Node]]]:
    """Build per-layer dense-region node lists for forward and backward.

    Dense region = the attention portion of a transformer block. For MoE
    layers, generic ``layers.N`` boundary nodes, ffn_norm, and moe nodes are
    deliberately excluded so FSDP launches do not overlap with MoE token
    exchange. For dense-only layers (no MoE), the entire block is dense.
    """
    regions: dict[int, dict[str, list[fx.Node]]] = {
        i: {"fwd_dense": [], "bwd_dense": []} for i in range(n_layers)
    }
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn:
            continue
        layer_id = _layer_id_from_fqn(fqn)
        if layer_id is None or layer_id not in regions:
            continue
        if layer_id in moe_layer_ids and not _is_moe_layer_dense_fqn(fqn):
            continue
        if not _is_dense_region_target_node(node):
            continue
        key = "bwd_dense" if _is_backward_node(node) else "fwd_dense"
        regions[layer_id][key].append(node)
    return regions


def _build_top_dense_regions(gm: torch.fx.GraphModule) -> dict[str, list[fx.Node]]:
    """Build dense compute regions for norm/lm_head outside transformer blocks."""
    regions: dict[str, list[fx.Node]] = {"fwd_dense": [], "bwd_dense": []}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn or not _is_top_level_param_fqn(fqn):
            continue
        if not _is_dense_region_target_node(node):
            continue
        key = "bwd_dense" if _is_backward_or_recomputed_node(node) else "fwd_dense"
        regions[key].append(node)
    return regions


def _read_fsdp_bucket_meta(node: fx.Node) -> tuple[tuple[str, ...], str] | None:
    meta = node.meta.get(_FSDP_BUCKET_META)
    if not isinstance(meta, dict):
        return None
    plan_fqns = meta.get("plan_fqns")
    direction = meta.get("direction")
    if not isinstance(plan_fqns, tuple) or direction not in {"fwd", "bwd"}:
        return None
    return plan_fqns, direction


def _collect_bucketed_fsdp_comms(
    gm: torch.fx.GraphModule,
    *,
    n_layers: int,
) -> tuple[dict[int, dict[str, list[tuple[fx.Node, fx.Node]]]], list[_FsdpComm]]:
    """Collect bucketed FSDP AG and RS (launch, wait) pairs per layer and direction.

    Returns ``({layer_id: {"fwd_ag": [...], "bwd_ag": [...], "bwd_rs": [...]}},
    all_comms)``. The per-layer view is kept for validation and the historical
    transformer-block placement rules; ``all_comms`` retains non-transformer
    bucket provenance so edge buckets can be placed without user-walking
    heuristics.
    """
    all_comms: list[_FsdpComm] = []
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    first_backward_pos = min(
        (order[node] for node in gm.graph.nodes if _is_backward_node(node)),
        default=None,
    )

    def _infer_fqn_and_direction(
        launch: fx.Node, wait: fx.Node
    ) -> tuple[str | None, bool]:
        def _nearest_from(starts: Iterable[fx.Node]) -> tuple[str | None, bool]:
            frontier = list(starts)
            visited: set[fx.Node] = set()
            while frontier:
                next_frontier: list[fx.Node] = []
                candidates: list[tuple[str, bool]] = []
                for n in frontier:
                    if n in visited or n.op == "output":
                        continue
                    visited.add(n)
                    candidate = n.meta.get("custom", {}).get(_MODULE_FQN)
                    if candidate:
                        candidates.append(
                            (candidate, _is_backward_or_recomputed_node(n))
                        )
                    next_frontier.extend(n.users)
                if candidates:
                    fqn = candidates[0][0]
                    return fqn, any(is_bwd for _, is_bwd in candidates)
                frontier = next_frontier
            return None, False

        # The wait use site is the semantic owner of a bucketed AG.  Do not
        # classify direction by arbitrarily walking to transitive backward
        # users; a forward parameter use naturally feeds backward later.
        fqn, is_bwd = _nearest_from(wait.users)
        if fqn is not None:
            if first_backward_pos is not None and order[wait] >= first_backward_pos:
                is_bwd = True
            return fqn, is_bwd
        fqn = wait.meta.get("custom", {}).get(_MODULE_FQN)
        if fqn is not None:
            is_bwd = _is_backward_or_recomputed_node(wait)
            if first_backward_pos is not None and order[wait] >= first_backward_pos:
                is_bwd = True
            return fqn, is_bwd
        fqn = launch.meta.get("custom", {}).get(_MODULE_FQN)
        if fqn is not None:
            is_bwd = _is_backward_or_recomputed_node(launch)
            if first_backward_pos is not None and order[wait] >= first_backward_pos:
                is_bwd = True
            return fqn, is_bwd
        return None, False

    comms: dict[int, dict[str, list[tuple[fx.Node, fx.Node]]]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        is_ag = is_all_gather(node)
        is_rs = node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
        if not is_ag and not is_rs:
            continue
        wait_nodes = [user for user in node.users if is_wait_tensor(user)]
        if not wait_nodes:
            continue
        wait = wait_nodes[0]
        bucket_meta = _read_fsdp_bucket_meta(node) or _read_fsdp_bucket_meta(wait)
        if bucket_meta is not None:
            plan_fqns, direction = bucket_meta
        else:
            # Fallback for tests and legacy graphs. Bucketed launches created
            # by this stack should carry _FSDP_BUCKET_META.
            fqn, is_bwd = _infer_fqn_and_direction(node, wait)
            if not fqn:
                continue
            plan_fqns = (fqn,)
            direction = "bwd" if is_bwd else "fwd"

        logical_index = _fsdp_bucket_logical_index(plan_fqns, n_layers=n_layers)
        kind = "ag" if is_ag else "rs"
        all_comms.append(
            _FsdpComm(
                launch=node,
                wait=wait,
                kind=kind,
                direction=direction,
                plan_fqns=plan_fqns,
                logical_index=logical_index,
            )
        )

        if logical_index is None or logical_index >= n_layers:
            continue
        layer_id = logical_index
        comms.setdefault(layer_id, {"fwd_ag": [], "bwd_ag": [], "bwd_rs": []})
        if kind == "ag" and direction == "fwd":
            comms[layer_id]["fwd_ag"].append((node, wait))
        elif kind == "ag" and direction == "bwd":
            comms[layer_id]["bwd_ag"].append((node, wait))
        elif kind == "rs" and direction == "bwd":
            comms[layer_id]["bwd_rs"].append((node, wait))
    logger.info(
        "Collected FSDP comms for dense-region scheduling: fwd_ag=%d bwd_ag=%d bwd_rs=%d",
        sum(len(layer_comms["fwd_ag"]) for layer_comms in comms.values()),
        sum(len(layer_comms["bwd_ag"]) for layer_comms in comms.values()),
        sum(len(layer_comms["bwd_rs"]) for layer_comms in comms.values()),
    )
    return comms, all_comms


def _has_waited_fsdp_comm_launches(gm: torch.fx.GraphModule) -> bool:
    """Return whether the graph contains any waited FSDP AG/RS launch."""
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if not (
            is_all_gather(node)
            or node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            continue
        if any(is_wait_tensor(user) for user in node.users):
            return True
    return False


def _validate_transformer_block_bucket_counts(
    comms: dict[int, dict[str, list[tuple[fx.Node, fx.Node]]]],
    *,
    n_layers: int,
    expected_bucket_counts: dict[int, int],
) -> None:
    missing_expected = [
        layer_id
        for layer_id in range(n_layers)
        if layer_id not in expected_bucket_counts
    ]
    extra_expected = [
        layer_id
        for layer_id in expected_bucket_counts
        if layer_id not in range(n_layers)
    ]
    errors: list[str] = []
    if missing_expected:
        errors.append(f"missing expected counts for layers {missing_expected}")
    if extra_expected:
        errors.append(f"unexpected expected-count layers {sorted(extra_expected)}")

    empty_layer_comms: dict[str, list[tuple[fx.Node, fx.Node]]] = {
        "fwd_ag": [],
        "bwd_ag": [],
        "bwd_rs": [],
    }
    for layer_id in range(n_layers):
        expected = expected_bucket_counts.get(layer_id)
        if expected is None:
            continue
        layer_comms = comms.get(layer_id, empty_layer_comms)
        actual = {
            "fwd_ag": len(layer_comms["fwd_ag"]),
            "bwd_ag": len(layer_comms["bwd_ag"]),
            "bwd_rs": len(layer_comms["bwd_rs"]),
        }
        mismatches = {
            kind: count for kind, count in actual.items() if count != expected
        }
        if mismatches:
            errors.append(
                f"layer {layer_id}: expected {expected} buckets per kind, "
                f"got fwd_ag={actual['fwd_ag']} bwd_ag={actual['bwd_ag']} "
                f"bwd_rs={actual['bwd_rs']}"
            )

    if errors:
        raise ValueError(
            "FSDP dense-region scheduling requires one bucketed transformer-block "
            "collective per expected bucket in each direction/type:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def schedule_fsdp_comms_to_dense_regions_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    moe_layer_ids: frozenset[int],
    n_layers: int,
    transformer_bucket_counts_by_layer: dict[int, int] | None = None,
    strict: bool = False,
) -> torch.fx.GraphModule:
    """Schedule bucketed FSDP comms to overlap with dense (attention) regions.

    Physically moves AG/RS launch nodes in the FX graph so that Inductor
    emits them at the desired positions. Inductor's
    ``decide_global_ordering_of_comms`` chains collectives in FX graph
    order, and the stable topological sort preserves FX order as a
    tiebreaker, so FX graph order IS the execution order for comms.

    Forward (execution order i-1 -> i):
      AG(i) launch at start of dense_fwd(i-1), for i > 0.
      AG(i) wait stays before layer i's first param use (unchanged).
      AG(0) is left to the original bucketing schedule.

    Backward (execution order i+1 -> i -> i-1):
      AG(i) launch at start of dense_bwd(i+1), for i < N-1.
      AG(i) wait stays before layer i's first param use (unchanged).
      AG(N-1) launch at start of top-level backward dense compute when present.
      RS(i) launch in dense_bwd(i-1), for i > 0, after its gradient input is ready.
      RS(0) is left to the original bucketing schedule.
      RS waits and pure output-unpack users are sunk to the graph tail.

    Embedding buckets are left to the upstream bucketer. The top-level norm/lm_head
    bucket is treated as the edge after transformer block N-1.
    """
    del example_inputs
    regions = _build_layer_dense_regions(gm, n_layers, moe_layer_ids)
    top_regions = _build_top_dense_regions(gm)
    has_fsdp_comms = _has_waited_fsdp_comm_launches(gm)
    comms, all_comms = _collect_bucketed_fsdp_comms(gm, n_layers=n_layers)

    if not has_fsdp_comms:
        return gm

    if transformer_bucket_counts_by_layer is not None:
        _validate_transformer_block_bucket_counts(
            comms,
            n_layers=n_layers,
            expected_bucket_counts=transformer_bucket_counts_by_layer,
        )

    order = {node: i for i, node in enumerate(gm.graph.nodes)}

    _FSDP_INFRA_OPS = {
        torch.ops.bucketing._pre_bucket_all_gather.default,
        torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops.aten.slice.Tensor,
    }
    _FSDP_WAIT_OUTPUT_OPS = {
        operator.getitem,
        torch.ops.aten._to_copy.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.view.default,
        torch.ops.aten.view.dtype,
    }

    wait_closures_sunk = 0

    def _is_allowed_wait_tail_node(
        node: fx.Node,
        closure: set[fx.Node],
        wait: fx.Node,
    ) -> tuple[bool, str | None]:
        if node.target in _FSDP_WAIT_OUTPUT_OPS:
            return True, None
        if node.target is not torch.ops.aten.add_.Tensor:
            return False, f"{wait.name} has non-output user {node.name} ({node.target})"

        mutated_arg = node.args[0] if node.args else None
        if not isinstance(mutated_arg, fx.Node):
            return False, f"{node.name} has non-node mutated input"

        # Chunked loss can accumulate multiple reduce-scattered grad shards via
        # an in-place add chain before returning the detached final shard. That
        # chain is safe to sink only when nothing else reads the accumulator.
        external_users = [
            user
            for user in mutated_arg.users
            if user is not node and user not in closure and user.op != "output"
        ]
        if external_users:
            users = ", ".join(user.name for user in external_users[:3])
            return (
                False,
                f"{node.name} mutates {mutated_arg.name}, which has "
                f"non-tail users: {users}",
            )
        return True, None

    def _collect_launch_chain(launch: fx.Node) -> list[fx.Node]:
        """Collect the FSDP infrastructure chain rooted at ``launch``."""
        chain: list[fx.Node] = []
        work = [launch]
        visited: set[fx.Node] = set()
        while work:
            n = work.pop()
            if n in visited or n.op in ("placeholder", "get_attr"):
                continue
            if n.target not in _FSDP_INFRA_OPS:
                continue
            visited.add(n)
            chain.append(n)
            work.extend(n.all_input_nodes)
        return chain

    def _find_dense_target(
        launch: fx.Node, dense_region: list[fx.Node]
    ) -> fx.Node | None:
        """Pick the first dense node after the launch chain's external deps."""
        chain = set(_collect_launch_chain(launch))
        if not chain:
            return None
        min_pos = 0
        for n in chain:
            for inp in n.all_input_nodes:
                if inp not in chain:
                    min_pos = max(min_pos, order.get(inp, -1) + 1)
        for target in dense_region:
            if order[target] >= min_pos:
                return target
        return None

    def _sink_output_only_wait_closure(wait: fx.Node) -> tuple[bool, str | None]:
        """Sink an RS wait and tail-only output/grad-accum users."""
        closure: set[fx.Node] = set()
        work = [wait]
        while work:
            n = work.pop()
            if n in closure:
                continue
            if n is not wait:
                allowed, reason = _is_allowed_wait_tail_node(n, closure, wait)
                if not allowed:
                    return False, reason
            closure.add(n)
            for user in n.users:
                if user.op == "output" or user in closure:
                    continue
                work.append(user)
        output = next(node for node in gm.graph.nodes if node.op == "output")
        for n in sorted(closure, key=lambda node: order.get(node, 0)):
            output.prepend(n)
        return True, None

    def _move_chain_before(launch: fx.Node, target: fx.Node) -> tuple[bool, list[str]]:
        """Move an AG/RS launch and its FSDP infrastructure chain before ``target``.

        Only moves FSDP-specific infrastructure nodes. In particular, RS
        gradient producers such as casts or adds stay where autograd produced
        them; target selection ensures the launch moves after those producers.
        """
        target_pos = order[target]

        chain = _collect_launch_chain(launch)

        # Only move nodes whose inputs can remain before the insertion point
        # and whose existing users will not be stranded before the moved node.
        # This supports both pulling late RS launches earlier and delaying
        # upstream-prefetched AG launches into the intended dense region.
        movable = set(chain)
        reasons: list[str] = []
        changed = True
        while changed:
            changed = False
            for n in list(movable):
                for inp in n.all_input_nodes:
                    if inp not in movable and order.get(inp, -1) >= target_pos:
                        reasons.append(
                            f"{n.name} is pinned before {target.name}: "
                            f"input {inp.name} occurs at {order.get(inp, -1)}, "
                            f"target at {target_pos}"
                        )
                        movable.discard(n)
                        changed = True
                        break
                if n not in movable:
                    continue
                if order.get(n, -1) < target_pos:
                    for user in n.users:
                        if user not in movable and order.get(user, -1) < target_pos:
                            reasons.append(
                                f"{n.name} is pinned before {target.name}: "
                                f"user {user.name} occurs at {order.get(user, -1)}, "
                                f"target at {target_pos}"
                            )
                            movable.discard(n)
                            changed = True
                            break
        to_move = sorted(movable, key=lambda x: order.get(x, 0))

        if not to_move:
            return False, reasons

        # Prepend in ascending order: each successive prepend inserts
        # between the previously moved node and target, producing
        # correct topological order before target.
        for n in to_move:
            target.prepend(n)
        return True, []

    moved = 0
    # Collect all moves first, then apply.  Recompute order after each
    # successful move so stale positions don't cause violations.
    moves: list[tuple[fx.Node, fx.Node, fx.Node, str]] = []
    blockers: list[str] = []

    def _add_move(
        launch: fx.Node,
        wait: fx.Node,
        dense_region: list[fx.Node],
        description: str,
    ) -> None:
        if not dense_region:
            blockers.append(f"{description}: no dense target region")
            return
        target = _find_dense_target(launch, dense_region)
        if target is None:
            blockers.append(f"{description}: launch inputs are after dense region")
            return
        moves.append((launch, wait, target, description))

    for layer_id, layer_comms in sorted(comms.items()):
        for ag_launch, _ag_wait in layer_comms["fwd_ag"]:
            prev = layer_id - 1
            if prev >= 0 and regions[prev]["fwd_dense"]:
                _add_move(
                    ag_launch,
                    _ag_wait,
                    regions[prev]["fwd_dense"],
                    f"fwd AG layer {layer_id} -> dense_fwd layer {prev}",
                )
        for ag_launch, _ag_wait in layer_comms["bwd_ag"]:
            next_layer = layer_id + 1
            if next_layer < n_layers and regions[next_layer]["bwd_dense"]:
                _add_move(
                    ag_launch,
                    _ag_wait,
                    regions[next_layer]["bwd_dense"],
                    f"bwd AG layer {layer_id} -> dense_bwd layer {next_layer}",
                )
        for rs_launch, _rs_wait in layer_comms["bwd_rs"]:
            prev = layer_id - 1
            if prev >= 0 and regions[prev]["bwd_dense"]:
                _add_move(
                    rs_launch,
                    _rs_wait,
                    regions[prev]["bwd_dense"],
                    f"bwd RS layer {layer_id} -> dense_bwd layer {prev}",
                )

    for comm in all_comms:
        if comm.logical_index != n_layers:
            continue
        if comm.kind == "ag" and comm.direction == "fwd" and n_layers > 0:
            _add_move(
                comm.launch,
                comm.wait,
                regions[n_layers - 1]["fwd_dense"],
                "fwd AG top-level bucket -> dense_fwd last transformer block",
            )
        elif comm.kind == "rs" and comm.direction == "bwd" and n_layers > 0:
            _add_move(
                comm.launch,
                comm.wait,
                regions[n_layers - 1]["bwd_dense"],
                "bwd RS top-level bucket -> dense_bwd last transformer block",
            )

    if n_layers > 0 and top_regions["bwd_dense"]:
        for ag_launch, ag_wait in comms.get(n_layers - 1, {}).get("bwd_ag", []):
            _add_move(
                ag_launch,
                ag_wait,
                top_regions["bwd_dense"],
                "bwd AG last transformer block -> top-level backward dense",
            )

    if blockers and strict:
        raise ValueError(
            "Could not schedule all interior FSDP comm launches to dense regions:\n"
            + "\n".join(f"- {blocker}" for blocker in blockers)
        )

    failed_moves: list[str] = []
    for comm in all_comms:
        if comm.kind != "rs":
            continue
        moved_wait, reason = _sink_output_only_wait_closure(comm.wait)
        wait_closures_sunk += int(moved_wait)
        if not moved_wait and strict:
            failed_moves.append(
                f"RS wait sink for {comm.plan_fqns!r} {comm.wait.name}: {reason}"
            )

    for launch, _wait, target, description in moves:
        # Recompute order before each move to account for prior moves.
        order = {node: i for i, node in enumerate(gm.graph.nodes)}
        moved_chain, reasons = _move_chain_before(launch, target)
        if moved_chain:
            moved += 1
        elif strict:
            reason = (
                "; ".join(reasons[:3]) if reasons else "no movable launch chain nodes"
            )
            failed_moves.append(f"{description}: {reason}")

    if failed_moves:
        raise ValueError(
            "Could not move FSDP comm launches to selected dense targets:\n"
            + "\n".join(f"- {move}" for move in failed_moves)
        )

    if moved or wait_closures_sunk:
        gm.graph.lint()
        gm.recompile()
    logger.info("schedule_fsdp_comms_to_dense_regions: moved %d comm chains", moved)
    return gm
