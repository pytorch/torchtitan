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
    force_joint_graph: bool = False,
) -> torch.fx.GraphModule:
    """Tag all saved activations eligible for CPU offloading.

    Two modes of operation:

    **Without SAC** (no recompute tags): Sets
    ``node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD`` and the
    caller inserts ao:: ops in the joint graph.

    **With SAC** (MUST_SAVE / PREFER_RECOMPUTE tags present): Sets
    ``node.meta["should_offload"] = True`` on MUST_SAVE nodes and enables
    ``config.enable_activation_offloading``. The partitioner's built-in
    offloading then inserts offload/reload ops AFTER the fwd/bwd split,
    so recomputed ops in backward naturally receive reloaded tensors.

    In both modes, the last layer is skipped (its activations are consumed
    immediately in backward).

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).

    Returns:
        The GraphModule with eligible nodes tagged for offload.
    """
    forward_nodes, _ = _classify_forward_backward(gm)
    sac_active = not force_joint_graph and _detect_sac_active(gm)

    # Find all layer IDs to determine the last layer
    all_layer_ids: set[int] = set()
    for node in gm.graph.nodes:
        lid = _get_layer_id(node)
        if lid != _NOT_IN_LAYERS:
            all_layer_ids.add(lid)
    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else _NOT_IN_LAYERS

    tagged = 0
    total_bytes = 0
    delegated = 0
    delegated_bytes = 0

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
            # SAC coexistence: delegate MUST_SAVE nodes to the partitioner's
            # built-in offloading, which inserts offload/reload AFTER the
            # fwd/bwd split. Recomputed ops in backward naturally receive
            # the reloaded tensor, avoiding the dealloc-vs-recompute conflict.
            if existing == CheckpointPolicy.MUST_SAVE:
                node.meta["should_offload"] = True
                node.meta["offload_group"] = layer_id
                delegated += 1
                delegated_bytes += _tensor_bytes(node.meta["val"])
            continue

        if force_joint_graph and existing == CheckpointPolicy.MUST_SAVE:
            # Post-remat: SAC already resolved recomputation. MUST_SAVE nodes
            # are the surviving activations — safe to offload on the joint graph.
            node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
            tagged += 1
            total_bytes += _tensor_bytes(node.meta["val"])
            continue

        # Non-SAC path: tag for joint-graph ao:: insertion
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

    if sac_active:
        logger.info(
            f"CPU offload (SAC coexistence): marked {delegated} MUST_SAVE "
            f"activations for partitioner-level offloading "
            f"({delegated_bytes / 1024 / 1024:.2f} MB)"
        )
    else:
        logger.info(
            f"CPU offload: tagged {tagged} activations for offload "
            f"({total_bytes / 1024 / 1024:.2f} MB)"
        )
    return gm


# ============================================================
# Graph passes
# ============================================================


def _sink_forward_waits_per_group(
    gm: torch.fx.GraphModule,
    wait_offload_map: dict[Node, Node],
    lookahead: int,
) -> None:
    """Sink forward wait_offload nodes to group boundaries.

    Group G's waits are moved to just after group (G + lookahead)'s last
    ao.offload node.  This gives nearly a full layer of compute-transfer
    overlap: the D2H copies from group G run in parallel with group
    G+lookahead's entire compute, so by the time the waits fire the
    transfers have finished.
    """
    node_to_idx: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # Collect wait_offload and offload nodes by layer group.
    wait_groups: dict[int, list[Node]] = {}
    offload_groups: dict[int, list[Node]] = {}

    for gpu_node, wait_node in wait_offload_map.items():
        layer_id = _get_layer_id(gpu_node)
        if layer_id == _NOT_IN_LAYERS:
            continue
        wait_groups.setdefault(layer_id, []).append(wait_node)

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.ao.offload.default:
            original = node.args[0]
            if isinstance(original, Node):
                layer_id = _get_layer_id(original)
                if layer_id != _NOT_IN_LAYERS:
                    offload_groups.setdefault(layer_id, []).append(node)

    if not wait_groups:
        return

    sorted_groups = sorted(wait_groups.keys())
    group_last_offload: dict[int, Node] = {
        g: max(nodes, key=lambda n: node_to_idx.get(n, 0))
        for g, nodes in offload_groups.items()
    }

    # Find the first backward node as fallback anchor for the last group's
    # waits.  Using output_node would place waits *after* backward nodes,
    # breaking reloads that reference them.
    first_bwd_node = None
    for n in gm.graph.nodes:
        if _is_backward_node(n):
            first_bwd_node = n
            break
    fallback_anchor = first_bwd_node or gm.graph.find_nodes(op="output")[0]

    moved = 0
    for idx, group in enumerate(sorted_groups):
        target_idx = min(idx + lookahead, len(sorted_groups) - 1)
        target_group = sorted_groups[target_idx]

        if target_idx > idx and target_group in group_last_offload:
            anchor = group_last_offload[target_group]
            for wait_node in wait_groups[group]:
                anchor.append(wait_node)
                moved += 1
        else:
            for wait_node in wait_groups[group]:
                fallback_anchor.prepend(wait_node)
                moved += 1

    logger.info(
        f"CPU offload: sunk {moved} forward waits across "
        f"{len(sorted_groups)} groups, lookahead={lookahead}"
    )


def apply_cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    sink_lookahead: int = 1,
    prefetch_lookahead: int = 1,
    cpu_budget_gb: float = 100.0,
) -> torch.fx.GraphModule:
    """Insert ao offload/reload/wait_tensor ops for nodes tagged ``MUST_CPU_OFFLOAD``.

    Reads ``node.meta["recompute"] is CheckpointPolicy.MUST_CPU_OFFLOAD`` (set by
    ``tag_all_offloadable_activations``) and inserts:
      Forward:  cpu = ao.offload(gpu_tensor)
                cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor)
                ao.dealloc(gpu_tensor, [forward_deps])
      Backward: gpu = ao.reload(cpu, device)
                gpu = ao.wait_tensor(gpu)
                replayed_views = replay view chain from gpu
    Then redirects backward consumers to use the reloaded/replayed tensors.

    View replay: views share storage with the base tensor. Instead of
    skipping dealloc for tensors with view descendants used in backward,
    we replay the view ops from the reloaded tensor and redirect backward
    consumers to the replayed versions. This allows deallocating ALL
    offloaded tensors' GPU storage in forward.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).

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
    dealloc_targets: list[Node] = []
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

            view_bwd = [u for u in view_node.users if _is_backward_node(u)]
            for user in view_bwd:
                user.replace_input_with(view_node, replay)

        dealloc_targets.append(node)

        logger.debug(
            f"CPU offload: offloading {node.name} "
            f"({_tensor_bytes(val) / 1024:.1f} KB, {val.shape})"
        )
        total_bytes += _tensor_bytes(val)

    # 4. Per-group forward wait sinking: sink each group's wait_offload to
    # N groups ahead so GPU memory is freed at group boundaries. Must happen
    # before dealloc insertion since dealloc deps reference wait nodes.
    if sink_lookahead > 0:
        _sink_forward_waits_per_group(gm, wait_offload_map, sink_lookahead)

    # 5. Insert dealloc ops to free GPU storage after all forward consumers.
    # After view replay above, all backward consumers have been redirected
    # to replayed views, so _get_storage_chain should find no backward users
    # and ALL offloaded tensors can be deallocated.
    node_positions: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}
    dealloc_plan: list[tuple[Node, list[Node], Node]] = []
    for gpu_node in dealloc_targets:
        chain_nodes, has_bwd = _get_storage_chain(gpu_node)
        if has_bwd:
            logger.warning(
                f"CPU offload: {gpu_node.name} still has backward users after "
                "view replay — skipping dealloc"
            )
            continue
        if not chain_nodes:
            continue
        tensor_deps = [
            n for n in chain_nodes if isinstance(n.meta.get("val"), torch.Tensor)
        ]
        last_consumer = max(chain_nodes, key=lambda n: node_positions[n])
        dealloc_plan.append((gpu_node, tensor_deps, last_consumer))

    for gpu_node, fwd_users, last_user in dealloc_plan:
        with gm.graph.inserting_after(last_user):
            dealloc_node = gm.graph.call_function(
                torch.ops.ao.dealloc.default,
                args=(gpu_node, fwd_users),
            )
            dealloc_node.meta["val"] = None

    # 6. Per-group backward reload prefetch.
    if prefetch_lookahead > 0:
        prefetch_offloads(gm, prefetch_lookahead)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"CPU offload: offloaded {len(offloadable)} tensors "
        f"({total_bytes / 1024 / 1024:.2f} MB), "
        f"deallocated {len(dealloc_plan)} in forward"
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
    prefetch_n_layers: int = 0,
    force_joint_graph: bool = False,
) -> torch.fx.GraphModule:
    """Two-phase CPU offload pass: tag eligible activations, then insert ops.

    This is the top-level entry point conforming to the graph pass signature.

    **Without SAC**: Tags activations with ``MUST_CPU_OFFLOAD`` and inserts
    ao:: offload/reload/wait ops directly in the joint graph.

    **With SAC**: Tags MUST_SAVE activations with ``should_offload`` and
    enables ``config.enable_activation_offloading``, delegating offload/reload
    insertion to the partitioner. This works because the partitioner inserts
    ops AFTER the fwd/bwd split, so recomputed backward ops naturally
    receive reloaded tensors.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by pass interface).
        prefetch_n_layers: When > 0, move ao.reload nodes this many layers
            earlier in the backward graph to overlap H2D with compute.

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    tag_all_offloadable_activations(gm, force_joint_graph=force_joint_graph)

    has_should_offload = not force_joint_graph and any(
        node.meta.get("should_offload", False) for node in gm.graph.nodes
    )
    if has_should_offload:
        # Reset the pytorch-side cumulative offload counter so the
        # partitioner's per-subgraph budget tracking starts fresh.
        import torch._functorch._activation_offloading.activation_offloading as _ao_mod
        import torch._functorch.config as functorch_config

        _ao_mod._cumulative_offload_bytes = 0

        sink_la = int(os.environ.get("OFFLOAD_SINK_LOOKAHEAD", "1"))
        prefetch_la = int(os.environ.get("OFFLOAD_PREFETCH_LOOKAHEAD", "1"))
        use_dealloc = os.environ.get("OFFLOAD_DEALLOC", "0") == "1"

        cpu_mem_budget = float(os.environ.get("OFFLOAD_CPU_MEMORY_BUDGET_GB", "100"))

        functorch_config.enable_activation_offloading = True
        functorch_config.activation_offload_separate_stream = True
        functorch_config.activation_offload_sink_wait = sink_la == 0
        functorch_config.activation_offload_sink_wait_group_lookahead = sink_la
        functorch_config.activation_reload_prefetch = prefetch_la == 0
        functorch_config.activation_reload_prefetch_group_lookahead = prefetch_la
        functorch_config.activation_offload_dealloc = use_dealloc
        functorch_config.activation_offload_cpu_memory_budget_gb = cpu_mem_budget
        logger.info(
            f"Enabled partitioner-level activation offloading for SAC coexistence "
            f"(sink_la={sink_la}, prefetch_la={prefetch_la}, dealloc={use_dealloc}, "
            f"cpu_mem_budget={cpu_mem_budget}GB)"
        )
    else:
        sink_la = int(os.environ.get("OFFLOAD_SINK_LOOKAHEAD", "1"))
        prefetch_la = int(os.environ.get("OFFLOAD_PREFETCH_LOOKAHEAD", "1"))
        cpu_mem_budget = float(os.environ.get("OFFLOAD_CPU_MEMORY_BUDGET_GB", "100"))
        apply_cpu_offload_pass(
            gm,
            example_inputs,
            sink_lookahead=sink_la,
            prefetch_lookahead=prefetch_la,
            cpu_budget_gb=cpu_mem_budget,
        )

    return gm
