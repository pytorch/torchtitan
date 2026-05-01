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
    cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor, dep=last_consumer)
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
    for node, nbytes in candidates:
        if budget_bytes > 0 and total_bytes + nbytes > budget_bytes:
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
        tagged += 1
        total_bytes += nbytes

    sizes_mb = sorted([nb / (1024 * 1024) for _, nb in candidates[:tagged]])
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
                cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor, dep=last_consumer)
      Backward: gpu = ao.reload(cpu, device)
                gpu = ao.wait_tensor(gpu)
    Then redirects backward consumers to use the reloaded tensors.

    GPU storage is freed inside ``ao.wait_tensor`` when ``keepalive`` is
    provided: after the D2H sync completes, the keepalive tensor's storage
    is released back to the allocator. The ``dep`` argument creates an
    explicit scheduling dependency that prevents graph reordering passes
    from moving the wait before the last forward consumer.

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
        node_custom = node.meta.get("custom", {})

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
            reload_node.meta["custom"] = dict(node_custom)

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
        prefetch_reloads(gm, prefetch_lookahead)

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
    """Defer each forward wait_tensor N layers past the last consumer.

    Each forward wait_tensor synchronizes the D2H transfer and frees GPU
    storage. Deferring it past the last consumer gives extra layers of
    compute to overlap with the D2H copy.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        wait_offload_map: Mapping from offloaded node to its wait_tensor node.
        n_layers: Number of layers to defer past the last consumer.

    Returns the number of wait nodes deferred.
    """
    # Find the last forward compute node per layer (excluding AO ops).
    layer_last_fwd_node: dict[int, Node] = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and not _is_backward_node(n):
            lid = _get_layer_id(n)
            if lid != _NOT_IN_LAYERS and n.target not in _AO_OPS:
                layer_last_fwd_node[lid] = n

    # Build next-layer map for cross-layer deferral.
    sorted_layers = sorted(layer_last_fwd_node.keys())
    next_layer_map: dict[int, int] = {}
    for i in range(len(sorted_layers) - 1):
        next_layer_map[sorted_layers[i]] = sorted_layers[i + 1]

    deferred = 0
    for node, wait_node in wait_offload_map.items():
        layer_id = _get_layer_id(node)
        if layer_id == _NOT_IN_LAYERS:
            continue

        # If storage chain extends to a later layer (cross-layer views),
        # use the latest consumer layer as the base.
        chain_nodes, _ = _get_storage_chain(node)
        real_consumers = {n for n in chain_nodes if n.target not in _AO_OPS}
        consumer_layer = layer_id
        for c in real_consumers:
            cl = _get_layer_id(c)
            if cl != _NOT_IN_LAYERS:
                consumer_layer = max(consumer_layer, cl)

        # Defer N layers past consumer layer if possible.
        defer_layer = consumer_layer
        for _ in range(n_layers):
            defer_layer = next_layer_map.get(defer_layer, defer_layer)
        if defer_layer not in layer_last_fwd_node:
            continue

        anchor = layer_last_fwd_node[defer_layer]
        wait_node.args = (*wait_node.args[:2], anchor)
        anchor.append(wait_node)
        deferred += 1

    return deferred


def prefetch_reloads(
    gm: torch.fx.GraphModule,
    n_layers: int,
) -> None:
    """Move ao.reload nodes N layers earlier in the backward for prefetching.

    For each ao.reload serving layer L's backward, moves it to just before
    layer (L + n_layers)'s first backward node. The corresponding
    ao.wait_tensor stays in place, so synchronization still happens just
    before the data is needed. This overlaps the H2D transfer with N layers
    of backward compute.
    """
    reload_info: list[tuple[Node, int]] = []
    for node in gm.graph.nodes:
        if not (
            node.op == "call_function"
            and node.target == torch.ops.ao.reload.default
            and _is_backward_node(node)
        ):
            continue

        layer_id = _get_reload_layer(node)
        if layer_id != _NOT_IN_LAYERS:
            reload_info.append((node, layer_id))

    if not reload_info:
        return

    # Find first backward compute node per layer (skip ao ops so we
    # anchor to real compute, not previously-inserted reload/wait nodes).
    layer_first_bwd: dict[int, Node] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or not _is_backward_node(node):
            continue
        if node.target in _AO_OPS:
            continue
        lid = _get_layer_id(node)
        if lid != _NOT_IN_LAYERS and lid not in layer_first_bwd:
            layer_first_bwd[lid] = node

    max_layer = max(layer_first_bwd.keys())

    moved = 0
    for reload_node, layer_id in reload_info:
        target_layer = min(layer_id + n_layers, max_layer)
        if target_layer == layer_id or target_layer not in layer_first_bwd:
            continue

        layer_first_bwd[target_layer].prepend(reload_node)
        moved += 1

    if moved > 0:
        logger.info(
            f"CPU offload prefetch: moved {moved} reloads {n_layers} layer(s) ahead"
        )
