# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU offload pass for activation offloading in graph_trainer.

Uses upstream custom ops (ao::offload, ao::reload, ao::wait_tensor) from
``torch._functorch._activation_offloading.offload_ops`` for async GPU<->CPU
transfers with event-based synchronization, and graph passes that insert
them around saved activations to reduce GPU memory.

Offload pattern (forward):
    cpu = ao.offload(gpu_tensor)
    ... forward consumers use gpu_tensor ...
    cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor)
    # wait_tensor frees gpu_tensor's storage after D2H completes

Reload pattern (backward):
    gpu = ao.reload(cpu, device)
    gpu = ao.wait_tensor(gpu)
    ... backward consumers use gpu ...

This module works with the make_fx traced joint fwd+bwd graph, using
``meta["custom"]["module_fqn"]`` for layer boundaries and
``meta["autograd_backward"]`` to distinguish forward from backward nodes.
"""

import operator

import torch

# Import upstream custom ops for async activation offloading.
# Registering the ops is a side-effect of importing the module.
import torch._functorch._activation_offloading.offload_ops  # noqa: F401
import torch.fx
from torch.fx import Node
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
)
from torchtitan.tools.logging import logger

aten = torch.ops.aten


# ============================================================
# Node eligibility
# ============================================================

# View ops by overloadpacket -- matches all overloads (e.g. aten.view.default,
# aten.view.dtype). Views share storage with input, so offloading doesn't free
# the base tensor's memory.
_VIEW_OPS = frozenset(
    {
        aten.t,
        aten.transpose,
        aten.view,
        aten._unsafe_view,
        aten.reshape,
        aten.unsqueeze,
        aten.squeeze,
        aten.slice,
        aten.select,
        aten.expand,
        aten.permute,
        aten.as_strided,
        aten.alias,
        aten.split,
        aten.narrow,
        aten.unfold,
        aten.detach,
    }
)

_MIN_OFFLOAD_BYTES = 4096  # 4 KB minimum to justify offload overhead


def _get_aten_target(node: Node) -> object:
    """Get the overloadpacket for an aten op node, for overload-agnostic matching."""
    target = node.target
    if hasattr(target, "overloadpacket"):
        return target.overloadpacket
    return target


def _is_view(node: Node) -> bool:
    """Check if a node produces a view (aliased memory, not a new allocation)."""
    return _get_aten_target(node) in _VIEW_OPS


def _get_storage_chain(node: Node) -> tuple[set[Node], bool]:
    """Walk the view chain to find all nodes sharing this tensor's storage.

    Views alias the base tensor's storage, so any node reachable through view
    edges (and its non-view consumers) depends on the storage being alive.

    Returns:
        (chain_nodes, has_bwd) where chain_nodes is every forward
        ``call_function`` node reachable through the view chain (including
        non-view endpoints) and has_bwd is True if any node in the chain
        has un-redirected backward consumers.
    """
    chain_nodes: set[Node] = set()
    has_bwd = False

    def _walk(n: Node) -> None:
        nonlocal has_bwd
        for user in n.users:
            if user.op != "call_function":
                continue
            if _is_backward_node(user):
                has_bwd = True
                continue
            chain_nodes.add(user)
            if _is_view(user):
                _walk(user)

    _walk(node)
    return chain_nodes, has_bwd


def _is_collective_or_wait(node: Node) -> bool:
    """Check if a node is a distributed collective or wait op."""
    target = node.target
    ns = getattr(target, "namespace", None)
    return ns == "_c10d_functional"


def _tensor_is_contiguous(val: torch.Tensor) -> bool:
    """Check contiguity, handling symbolic FakeTensors gracefully."""
    try:
        return bool(val.is_contiguous())
    except Exception:
        # Symbolic shapes can't be evaluated statically; assume contiguous
        return True


def _tensor_bytes(val: torch.Tensor) -> int:
    """Get tensor size in bytes, handling symbolic FakeTensors gracefully."""
    try:
        nbytes = val.nelement() * val.element_size()
        # Force evaluation to a concrete int; symbolic expressions may
        # raise GuardOnDataDependentSymNode when compared with <.
        return int(nbytes)
    except Exception:
        # Symbolic shapes can't be evaluated; assume large enough
        return _MIN_OFFLOAD_BYTES


def _has_offloadable_tensor(node: Node) -> bool:
    """Check if a node produces a tensor worth offloading (large, contiguous)."""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return False
    if not _tensor_is_contiguous(val):
        return False
    if _tensor_bytes(val) < _MIN_OFFLOAD_BYTES:
        return False
    return True


def _can_offload_node(node: Node) -> bool:
    """Check if a node's output can be profitably offloaded to CPU.

    Excludes views (offloading doesn't free base tensor memory), small tensors
    (overhead exceeds benefit), non-contiguous tensors, and collective/wait ops.

    getitem nodes are allowed -- they unpack multi-output ops and produce
    real tensors, not views.
    """
    if node.op != "call_function":
        return False
    if node.target is operator.getitem:
        return _has_offloadable_tensor(node)
    if _is_view(node):
        return False
    if _is_collective_or_wait(node):
        return False
    return _has_offloadable_tensor(node)


def _has_recompute_consumer(node: Node) -> bool:
    """Check if any forward user (or transitive view user) is tagged for recomputation."""
    for user in node.users:
        if user.op != "call_function" or _is_backward_node(user):
            continue
        policy = user.meta.get("recompute")
        if policy in (
            CheckpointPolicy.MUST_RECOMPUTE,
            CheckpointPolicy.PREFER_RECOMPUTE,
        ):
            return True
        if _is_view(user) and _has_recompute_consumer(user):
            return True
    return False


# ============================================================
# Forward/backward node classification for make_fx traced graphs
# ============================================================


def _classify_forward_backward(
    gm: torch.fx.GraphModule,
) -> tuple[set[Node], set[Node]]:
    """Classify nodes in a make_fx traced joint graph as forward or backward.

    Uses ``autograd_backward`` metadata set by PyTorch's autograd engine during
    make_fx tracing.

    Returns:
        Tuple of (forward_nodes, backward_nodes).
    """
    forward_nodes: set[Node] = set()
    backward_nodes: set[Node] = set()

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "get_attr"):
            continue
        if _is_backward_node(node):
            backward_nodes.add(node)
        else:
            forward_nodes.add(node)

    return forward_nodes, backward_nodes


# ============================================================
# Tagging pass
# ============================================================


def tag_all_offloadable_activations(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    cpu_budget_gb: float = 100.0,
) -> torch.fx.GraphModule:
    """Tag saved activations eligible for CPU offloading.

    Sets ``node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD`` on
    eligible forward nodes. Nodes already tagged MUST_SAVE (by SAC) are
    offloaded; nodes tagged for recomputation are skipped. Without any
    prior tagging, any forward node with backward consumers is eligible.

    Heuristic: the last layer is skipped because its activations are
    consumed immediately in backward, so offloading adds overhead with no
    benefit. This may not hold for chunked loss where the last layer's
    activations are consumed later.

    Applies a CPU memory budget: tensors are selected largest-first until
    the budget is exhausted.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).
        cpu_budget_gb: Maximum CPU memory budget (GB per rank) for offloaded
            activations. Tensors are selected largest-first.

    Returns:
        The GraphModule with eligible nodes tagged for offload.
    """
    forward_nodes, _ = _classify_forward_backward(gm)

    # Find all layer IDs to determine the last layer
    all_layer_ids: set[int] = set()
    for node in gm.graph.nodes:
        lid = _get_layer_id(node)
        if lid != _NOT_IN_LAYERS:
            all_layer_ids.add(lid)
    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else _NOT_IN_LAYERS

    # Collect candidates with sizes for budget filtering.
    candidates: list[tuple[Node, int]] = []

    for node in gm.graph.nodes:
        if node not in forward_nodes:
            continue
        layer_id = _get_layer_id(node)
        if layer_id == _NOT_IN_LAYERS or layer_id == last_layer_id:
            continue
        if not _can_offload_node(node):
            continue

        existing = node.meta.get("recompute")
        if existing == CheckpointPolicy.MUST_SAVE:
            pass
        elif existing in (
            CheckpointPolicy.MUST_RECOMPUTE,
            CheckpointPolicy.PREFER_RECOMPUTE,
        ):
            continue
        else:
            if _has_recompute_consumer(node):
                continue
            _, has_bwd = _get_storage_chain(node)
            if not has_bwd:
                continue

        candidates.append((node, _tensor_bytes(node.meta["val"])))

    # Apply budget: select largest-first until exhausted.
    budget_bytes = cpu_budget_gb * (1024**3)
    candidates.sort(key=lambda t: t[1], reverse=True)
    tagged = 0
    total_bytes = 0
    selected_sizes: list[int] = []
    for node, nbytes in candidates:
        if budget_bytes > 0 and total_bytes + nbytes > budget_bytes:
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
        tagged += 1
        total_bytes += nbytes
        selected_sizes.append(nbytes)

    sizes_mb = sorted([nb / (1024 * 1024) for nb in selected_sizes])
    if sizes_mb:
        logger.info(
            f"CPU offload: tagged {tagged}/{len(candidates)} activations for offload "
            f"({total_bytes / 1024 / 1024:.2f} MB), "
            f"sizes: min={sizes_mb[0]:.1f} MB, median={sizes_mb[len(sizes_mb) // 2]:.1f} MB, "
            f"max={sizes_mb[-1]:.1f} MB"
        )
    else:
        logger.info("CPU offload: no activations tagged for offload")
    return gm


# ============================================================
# Graph passes
# ============================================================


def apply_cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    prefetch_lookahead: int = 1,
    defer_n_layers: int = 1,
) -> torch.fx.GraphModule:
    """Insert ao offload/reload/wait_tensor ops for nodes tagged ``MUST_CPU_OFFLOAD``.

    Reads ``node.meta["recompute"] is CheckpointPolicy.MUST_CPU_OFFLOAD`` (set by
    ``tag_all_offloadable_activations``) and inserts:
      Forward:  cpu = ao.offload(gpu_tensor)
                cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor)
      Backward: gpu = ao.reload(cpu, device)
                gpu = ao.wait_tensor(gpu)
    Then redirects backward consumers to use the reloaded tensors.

    GPU storage is freed inside ``ao.wait_tensor`` when ``keepalive`` is
    provided: after the D2H sync completes, the keepalive tensor's storage
    is released back to the allocator. The ``defer_offload_waits`` pass
    physically moves each wait after the last consumer via
    ``anchor.append()``, which is preserved by the stable topological
    sort in the bucketing pass.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).
        prefetch_lookahead: Move ao.reload nodes this many layers earlier in
            the backward graph to overlap H2D with compute.
        defer_n_layers: Defer forward wait_tensor ops this many layers past
            the last consumer to overlap D2H with compute.

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    # 1. Collect nodes tagged for offload with their backward consumers.
    offloadable: list[tuple[Node, list[Node]]] = []
    for node in gm.graph.nodes:
        if node.meta.get("recompute") is not CheckpointPolicy.MUST_CPU_OFFLOAD:
            continue
        bwd_users = [u for u in node.users if _is_backward_node(u)]
        if not bwd_users:
            continue
        offloadable.append((node, bwd_users))

    if not offloadable:
        return gm

    # 2. Build position index for ordering queries.
    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # 3. Insert offload/reload/wait_tensor ops.
    total_bytes = 0
    wait_offload_map: dict[Node, Node] = {}

    for node, bwd_users in offloadable:
        val = node.meta.get("val")
        assert (
            val is not None
        ), f"Node {node.name} tagged for offload has no 'val' metadata"

        device = val.device
        # --- Forward: async GPU->CPU offload right after production ---
        with gm.graph.inserting_after(node):
            offload_node = gm.graph.call_function(
                torch.ops.ao.offload.default,
                args=(node,),
            )
            offload_node.meta["val"] = val.to(torch.device("cpu"))

        with gm.graph.inserting_after(offload_node):
            wait_offload_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(offload_node, node),
            )
            wait_offload_node.meta["val"] = offload_node.meta["val"]

        wait_offload_map[node] = wait_offload_node

        # --- Backward: async CPU->GPU reload before earliest consumer ---
        first_consumer = min(bwd_users, key=lambda n: node_to_index[n])

        with gm.graph.inserting_before(first_consumer):
            reload_node = gm.graph.call_function(
                torch.ops.ao.reload.default,
                args=(wait_offload_node, device),
            )
            reload_node.meta["val"] = val
            reload_node.meta["autograd_backward"] = True
            reload_node.meta["custom"] = dict(node.meta.get("custom", {}))

        with gm.graph.inserting_before(first_consumer):
            wait_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(reload_node,),
            )
            wait_node.meta["val"] = val
            wait_node.meta["autograd_backward"] = True

        for user in bwd_users:
            user.replace_input_with(node, wait_node)

        # Store mapping so the remat pass can redirect recomputed nodes
        # to the reloaded tensor instead of the freed forward tensor.
        node.meta["cpu_offload_reload_node"] = wait_node

        logger.debug(
            f"CPU offload: offloading {node.name} "
            f"({_tensor_bytes(val) / 1024:.1f} KB, {val.shape})"
        )
        total_bytes += _tensor_bytes(val)

    # 4b. Defer forward waits to maximize D2H/compute overlap.
    if defer_n_layers > 0:
        deferred = defer_offload_waits(gm, wait_offload_map, defer_n_layers)
    else:
        deferred = 0

    # 5. Per-group backward reload prefetch.
    if prefetch_lookahead > 0:
        prefetched = prefetch_reloads(gm, prefetch_lookahead)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"CPU offload: offloaded {len(offloadable)} tensors "
        f"({total_bytes / 1024 / 1024:.2f} MB), "
        f"deferred {deferred} forward waits"
    )
    return gm


def _get_reload_layer(reload_node: Node) -> int:
    """Get the layer ID of an offloaded activation from the reload node's metadata."""
    return _get_layer_id(reload_node)


_AO_OPS = {
    torch.ops.ao.offload.default,
    torch.ops.ao.reload.default,
    torch.ops.ao.wait_tensor.default,
}


def defer_offload_waits(
    gm: torch.fx.GraphModule,
    wait_offload_map: dict[Node, Node],
    n_layers: int = 1,
) -> int:
    """Defer each forward wait_tensor N regions past the last consumer.

    Each forward wait_tensor synchronizes the D2H transfer and frees GPU
    storage. Deferring it past the last consumer gives extra compute
    to overlap with the D2H copy.

    Regions are contiguous groups of forward compute nodes with the same
    layer_id. Non-layer segments (embeddings, lm_head, loss) form their
    own regions, so waits can be deferred across layer boundaries.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        wait_offload_map: Mapping from offloaded node to its wait_tensor node.
        n_layers: Number of regions to defer past the last consumer.

    Returns the number of wait nodes deferred.
    """
    # Build forward region anchors (last compute node per contiguous region)
    # and a mapping from each forward compute node to its region index.
    # Stop at the first backward node: non-backward utility ops (e.g.
    # _get_submesh) can appear after the backward section and must not
    # be treated as forward anchors.
    fwd_anchors: list[Node] = []
    node_to_region_idx: dict[Node, int] = {}
    current_layer = None
    region_idx = -1
    last_node = None

    for n in gm.graph.nodes:
        if n.op == "call_function" and _is_backward_node(n):
            break
        if n.op != "call_function":
            continue
        if n.target in _AO_OPS:
            continue
        lid = _get_layer_id(n)
        if lid != current_layer:
            if last_node is not None:
                fwd_anchors.append(last_node)
            region_idx += 1
            current_layer = lid
        node_to_region_idx[n] = region_idx
        last_node = n

    if last_node is not None:
        fwd_anchors.append(last_node)

    deferred = 0
    for node, wait_node in wait_offload_map.items():
        consumer_idx = node_to_region_idx.get(node)
        if consumer_idx is None:
            continue

        # If storage chain extends to a later region (cross-layer views),
        # use the latest consumer region.
        chain_nodes, _ = _get_storage_chain(node)
        real_consumers = {n for n in chain_nodes if n.target not in _AO_OPS}
        for c in real_consumers:
            idx = node_to_region_idx.get(c)
            if idx is not None:
                consumer_idx = max(consumer_idx, idx)

        # Defer N regions past the consumer.
        target_idx = min(consumer_idx + n_layers, len(fwd_anchors) - 1)
        anchor = fwd_anchors[target_idx]
        anchor.append(wait_node)
        deferred += 1

    return deferred


def prefetch_reloads(
    gm: torch.fx.GraphModule,
    n_layers: int,
) -> int:
    """Move ao.reload nodes N layers earlier in the backward for prefetching.

    Counts by layer transitions rather than raw regions, because
    _NOT_IN_LAYERS gradient accumulation nodes are interleaved between
    layer-specific backward nodes, creating many micro-regions per layer.

    Non-layer segments (loss backward, lm_head backward) are reachable
    when the target goes past all layers — e.g. prefetching the last
    layer's reloads into loss backward.

    The corresponding ao.wait_tensor stays in place, so synchronization
    still happens just before the data is needed.
    """
    # Build backward region anchors: first compute node per contiguous region.
    bwd_anchors: list[Node] = []
    layer_to_bwd_idx: dict[int, int] = {}
    current_layer = None

    for n in gm.graph.nodes:
        if n.op != "call_function" or not _is_backward_node(n):
            continue
        if n.target in _AO_OPS:
            continue
        lid = _get_layer_id(n)
        if lid != current_layer:
            bwd_anchors.append(n)
            if lid != _NOT_IN_LAYERS and lid not in layer_to_bwd_idx:
                layer_to_bwd_idx[lid] = len(bwd_anchors) - 1
            current_layer = lid

    if not bwd_anchors:
        return 0

    # Build layer order for counting by layer transitions.
    # Sorted by bwd_idx ascending = backward execution order (descending layer ID).
    bwd_layer_order = sorted(
        layer_to_bwd_idx.keys(), key=lambda lid: layer_to_bwd_idx[lid]
    )
    layer_pos_in_bwd = {lid: i for i, lid in enumerate(bwd_layer_order)}

    reload_info: list[tuple[Node, int]] = []
    for node in gm.graph.nodes:
        if not (
            node.op == "call_function"
            and node.target == torch.ops.ao.reload.default
            and _is_backward_node(node)
        ):
            continue
        layer_id = _get_reload_layer(node)
        if layer_id in layer_to_bwd_idx:
            reload_info.append((node, layer_id))

    if not reload_info:
        return 0

    moved = 0
    for reload_node, layer_id in reload_info:
        pos = layer_pos_in_bwd[layer_id]
        target_layer_pos = pos - n_layers

        if target_layer_pos < 0:
            # Gone past all layers — use the first backward region
            # (e.g. loss_bw), enabling cross-layer-boundary prefetch.
            target_idx = 0
        else:
            target_layer = bwd_layer_order[target_layer_pos]
            target_idx = layer_to_bwd_idx[target_layer]

        current_idx = layer_to_bwd_idx[layer_id]
        if target_idx >= current_idx:
            continue

        bwd_anchors[target_idx].prepend(reload_node)
        moved += 1

    if moved > 0:
        logger.info(
            f"CPU offload prefetch: moved {moved} reloads {n_layers} layer(s) ahead"
        )
    return moved
