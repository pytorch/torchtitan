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
import os

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


def _collect_view_tree(node: Node) -> list[Node]:
    """BFS collecting all forward view descendants in topological order.

    Views share storage with the base tensor, so they must be replayed from
    a reloaded tensor to allow deallocating the original GPU storage.
    """
    result: list[Node] = []
    queue = [node]
    visited = {node}
    while queue:
        current = queue.pop(0)
        for user in current.users:
            if user in visited or user.op != "call_function":
                continue
            if _is_backward_node(user) or not _is_view(user):
                continue
            visited.add(user)
            result.append(user)
            queue.append(user)
    return result


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


def _detect_sac_active(gm: torch.fx.GraphModule) -> bool:
    """Check if SAC has already tagged nodes in the graph."""
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        policy = node.meta.get("recompute")
        if policy in (
            CheckpointPolicy.MUST_SAVE,
            CheckpointPolicy.PREFER_RECOMPUTE,
            CheckpointPolicy.MUST_RECOMPUTE,
        ):
            return True
    return False


def tag_all_offloadable_activations(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Tag all saved activations eligible for CPU offloading.

    Sets ``node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD`` on
    eligible forward nodes. When SAC is active, only MUST_SAVE nodes are
    tagged (the surviving activations after recomputation decisions).
    Without SAC, any forward node with backward consumers is eligible.

    The last layer is skipped (its activations are consumed immediately
    in backward).

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).

    Returns:
        The GraphModule with eligible nodes tagged for offload.
    """
    forward_nodes, _ = _classify_forward_backward(gm)
    sac_active = _detect_sac_active(gm)

    # Find all layer IDs to determine the last layer
    all_layer_ids: set[int] = set()
    for node in gm.graph.nodes:
        lid = _get_layer_id(node)
        if lid != _NOT_IN_LAYERS:
            all_layer_ids.add(lid)
    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else _NOT_IN_LAYERS

    tagged = 0
    total_bytes = 0

    for node in gm.graph.nodes:
        if node not in forward_nodes:
            continue
        layer_id = _get_layer_id(node)
        if layer_id == _NOT_IN_LAYERS or layer_id == last_layer_id:
            continue
        if not _can_offload_node(node):
            continue

        existing = node.meta.get("recompute")

        if sac_active:
            # SAC already resolved recomputation. Only MUST_SAVE nodes are
            # the surviving activations — safe to offload.
            if existing != CheckpointPolicy.MUST_SAVE:
                continue
        else:
            if existing in (
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.PREFER_RECOMPUTE,
            ):
                continue
            if _has_recompute_consumer(node):
                continue
            _, has_bwd = _get_storage_chain(node)
            if not has_bwd:
                continue

        node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
        tagged += 1
        total_bytes += _tensor_bytes(node.meta["val"])

    logger.info(
        f"CPU offload: tagged {tagged} activations for offload "
        f"({total_bytes / 1024 / 1024:.2f} MB)"
    )
    return gm


# ============================================================
# Graph passes
# ============================================================


def _find_tensor_dep(node: Node, node_positions: dict[Node, int]) -> Node | None:
    """Find a tensor-valued node suitable for a scheduling dependency."""
    if isinstance(node.meta.get("val"), torch.Tensor):
        return node
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target is operator.getitem
            and isinstance(user.meta.get("val"), torch.Tensor)
            and not _is_backward_node(user)
        ):
            return user
    return None


def apply_cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    prefetch_lookahead: int = 1,
    cpu_budget_gb: float = 100.0,
) -> torch.fx.GraphModule:
    """Insert ao offload/reload/wait_tensor ops for nodes tagged ``MUST_CPU_OFFLOAD``.

    Reads ``node.meta["recompute"] is CheckpointPolicy.MUST_CPU_OFFLOAD`` (set by
    ``tag_all_offloadable_activations``) and inserts:
      Forward:  cpu = ao.offload(gpu_tensor)
                cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor, dep=last_consumer)
      Backward: gpu = ao.reload(cpu, device)
                gpu = ao.wait_tensor(gpu)
                replayed_views = replay view chain from gpu
    Then redirects backward consumers to use the reloaded/replayed tensors.

    GPU storage is freed inside ``ao.wait_tensor`` when ``keepalive`` is
    provided: after the D2H sync completes, the keepalive tensor's storage
    is released back to the allocator. The ``dep`` argument creates an
    explicit scheduling dependency that prevents graph reordering passes
    from moving the wait before the last forward consumer.

    Forward waits are repositioned after the last forward consumer of each
    GPU tensor's storage chain, maximizing D2H/compute overlap.

    View replay: views share storage with the base tensor. We replay the
    view ops from the reloaded tensor in backward and redirect backward
    consumers to the replayed versions.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).
        prefetch_lookahead: Move ao.reload nodes this many layers earlier in
            the backward graph to overlap H2D with compute.
        cpu_budget_gb: Maximum CPU memory budget for offloaded activations.

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    # 1. Collect nodes tagged for offload with their backward consumers
    # (direct and through view chain)
    offloadable: list[tuple[Node, list[Node], list[Node], list[Node]]] = []
    for node in gm.graph.nodes:
        if node.meta.get("recompute") is not CheckpointPolicy.MUST_CPU_OFFLOAD:
            continue
        direct_bwd = [u for u in node.users if _is_backward_node(u)]
        view_tree = _collect_view_tree(node)
        all_bwd = list(direct_bwd)
        for vn in view_tree:
            all_bwd.extend(u for u in vn.users if _is_backward_node(u))
        if not all_bwd:
            continue
        offloadable.append((node, direct_bwd, view_tree, all_bwd))

    if not offloadable:
        return gm

    # 2. Apply CPU memory budget: sort by size descending, accept until budget.
    budget_bytes = cpu_budget_gb * (1024**3)
    if budget_bytes > 0:
        offloadable.sort(key=lambda t: _tensor_bytes(t[0].meta["val"]), reverse=True)
        filtered = []
        cumulative = 0
        for entry in offloadable:
            nbytes = _tensor_bytes(entry[0].meta["val"])
            if cumulative + nbytes > budget_bytes:
                continue
            cumulative += nbytes
            filtered.append(entry)
        logger.info(
            f"CPU offload budget: selected {len(filtered)}/{len(offloadable)} "
            f"tensors ({cumulative / 1024**3:.2f} / {cpu_budget_gb:.1f} GB)"
        )
        offloadable = filtered

    if not offloadable:
        return gm

    # 3. Build position index for ordering queries (only used for pre-existing
    # nodes; newly inserted nodes never appear in all_bwd).
    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # 4. Insert offload/reload/wait_tensor + view replay
    total_bytes = 0
    wait_offload_map: dict[Node, Node] = {}

    for node, direct_bwd, view_tree, all_bwd in offloadable:
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
        first_consumer = min(all_bwd, key=lambda n: node_to_index[n])

        with gm.graph.inserting_before(first_consumer):
            reload_node = gm.graph.call_function(
                torch.ops.ao.reload.default,
                args=(wait_offload_node, device),
            )
            reload_node.meta["val"] = val
            reload_node.meta["autograd_backward"] = True
            reload_node.meta["partitioner_tag"] = "must_be_in_backward"

        with gm.graph.inserting_before(first_consumer):
            wait_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(reload_node,),
            )
            wait_node.meta["val"] = val
            wait_node.meta["autograd_backward"] = True
            wait_node.meta["partitioner_tag"] = "must_be_in_backward"

        # Redirect direct backward users to the reloaded tensor
        for user in direct_bwd:
            user.replace_input_with(node, wait_node)

        # Store mapping so the remat pass can redirect recomputed nodes
        # to the reloaded tensor instead of the freed forward tensor.
        node.meta["cpu_offload_reload_node"] = wait_node

        # Replay view tree in backward and redirect view backward users
        replay_map: dict[Node, Node] = {node: wait_node}
        insert_point = wait_node

        for view_node in view_tree:
            new_args = tuple(
                replay_map.get(a, a) if isinstance(a, Node) else a
                for a in view_node.args
            )
            with gm.graph.inserting_after(insert_point):
                replay = gm.graph.call_function(
                    view_node.target,
                    args=new_args,
                    kwargs=view_node.kwargs,
                )
                replay.meta["val"] = view_node.meta.get("val")
                replay.meta["autograd_backward"] = True
                replay.meta["partitioner_tag"] = "must_be_in_backward"

            replay_map[view_node] = replay
            insert_point = replay

            # Store mapping so the remat pass can redirect recomputed nodes
            # that access this view to the replayed backward version.
            view_node.meta["cpu_offload_reload_node"] = replay

            view_bwd = [u for u in view_node.users if _is_backward_node(u)]
            for user in view_bwd:
                user.replace_input_with(view_node, replay)

        logger.debug(
            f"CPU offload: offloading {node.name} "
            f"({_tensor_bytes(val) / 1024:.1f} KB, {val.shape})"
        )
        total_bytes += _tensor_bytes(val)

    # 4b. Move each forward wait to after the last forward consumer of
    # its GPU tensor. This maximizes D2H/compute overlap and lets
    # wait_tensor free GPU storage at the optimal point.
    _AO_OPS = {
        torch.ops.ao.offload.default,
        torch.ops.ao.reload.default,
        torch.ops.ao.wait_tensor.default,
    }
    node_positions: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}
    repositioned = 0
    for node, direct_bwd, view_tree, all_bwd in offloadable:
        wait_node = wait_offload_map[node]
        chain_nodes, has_bwd = _get_storage_chain(node)
        real_consumers = {n for n in chain_nodes if n.target not in _AO_OPS}
        if has_bwd or not real_consumers:
            continue
        last_consumer = max(real_consumers, key=lambda n: node_positions.get(n, 0))
        dep_node = _find_tensor_dep(last_consumer, node_positions)
        if dep_node is not None:
            wait_node.args = (*wait_node.args[:2], dep_node)
            dep_node.append(wait_node)
        else:
            last_consumer.append(wait_node)
        repositioned += 1

    # 5. Per-group backward reload prefetch.
    if prefetch_lookahead > 0:
        prefetch_offloads(gm, prefetch_lookahead)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"CPU offload: offloaded {len(offloadable)} tensors "
        f"({total_bytes / 1024 / 1024:.2f} MB), "
        f"repositioned {repositioned} forward waits"
    )
    return gm


def _get_reload_layer(reload_node: Node) -> int:
    """Get the layer ID of an offloaded activation from the forward op chain.

    Traces: ao.reload(fwd_wait) -> fwd_wait = ao.wait_tensor(offload, orig)
    -> offload = ao.offload(orig) -> orig has module_fqn.
    """
    fwd_wait = reload_node.args[0] if reload_node.args else None
    if not isinstance(fwd_wait, Node):
        return _NOT_IN_LAYERS
    offload = fwd_wait.args[0] if fwd_wait.args else None
    if not isinstance(offload, Node):
        return _NOT_IN_LAYERS
    original = offload.args[0] if offload.args else None
    if not isinstance(original, Node):
        return _NOT_IN_LAYERS
    return _get_layer_id(original)


def prefetch_offloads(
    gm: torch.fx.GraphModule,
    n_layers: int,
) -> None:
    """Move ao.reload nodes N layers earlier in the backward for prefetching.

    For each ao.reload serving layer L's backward, moves it to just before
    layer (L + n_layers)'s first backward wait node. The corresponding
    ao.wait_tensor stays in place, so synchronization still happens just
    before the data is needed. This overlaps the H2D transfer with N layers
    of backward compute.

    Layer IDs are determined from the forward op chain (ao.reload ->
    ao.wait_tensor -> ao.offload -> original forward node) rather than
    backward node metadata, since backward nodes may not carry module_fqn
    annotations in all tracing modes.
    """
    reload_info: list[tuple[Node, Node, int]] = []
    for node in gm.graph.nodes:
        if not (
            node.op == "call_function"
            and node.target == torch.ops.ao.reload.default
            and _is_backward_node(node)
        ):
            continue
        wait_node = next(
            (u for u in node.users if u.target == torch.ops.ao.wait_tensor.default),
            None,
        )
        if wait_node is None:
            continue

        layer_id = _get_reload_layer(node)
        if layer_id != _NOT_IN_LAYERS:
            reload_info.append((node, wait_node, layer_id))

    if not reload_info:
        return

    node_to_idx: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}
    reload_info.sort(key=lambda x: node_to_idx[x[1]])

    layer_first_wait: dict[int, Node] = {}
    for _reload, wait, lid in reload_info:
        if lid not in layer_first_wait:
            layer_first_wait[lid] = wait

    max_layer = max(layer_first_wait.keys())

    moved = 0
    for reload_node, _wait_node, layer_id in reload_info:
        target_layer = min(layer_id + n_layers, max_layer)
        if target_layer == layer_id or target_layer not in layer_first_wait:
            continue

        layer_first_wait[target_layer].prepend(reload_node)
        moved += 1

    if moved > 0:
        logger.info(
            f"CPU offload prefetch: moved {moved} reloads " f"{n_layers} layer(s) ahead"
        )


def cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    prefetch_n_layers: int = 1,
) -> torch.fx.GraphModule:
    """Tag eligible activations and insert ao:: offload/reload/wait ops.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by pass interface).
        prefetch_n_layers: Move ao.reload nodes this many layers earlier in
            the backward graph to overlap H2D with compute.

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    tag_all_offloadable_activations(gm)

    cpu_mem_budget = float(os.environ.get("OFFLOAD_CPU_MEMORY_BUDGET_GB", "100"))
    apply_cpu_offload_pass(
        gm,
        example_inputs,
        prefetch_lookahead=prefetch_n_layers,
        cpu_budget_gb=cpu_mem_budget,
    )

    return gm
