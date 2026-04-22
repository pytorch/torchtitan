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
``meta["custom"]["ac_region_id"]`` as layer boundaries and
``meta["autograd_backward"]`` to distinguish forward from backward nodes.
"""

import operator

import torch

# Import upstream custom ops for async activation offloading.
# Registering the ops is a side-effect of importing the module.
import torch._functorch._activation_offloading.offload_ops as offload_ops  # noqa: F401
import torch.fx
from torch.fx import Node
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import _AC_REGION_ID
from torchtitan.tools.logging import logger

aten = torch.ops.aten


def _is_backward_node(node: Node) -> bool:
    return node.meta.get("autograd_backward", False)


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
    }
)

# Collective/wait ops whose outputs should not be offloaded
_COLLECTIVE_OPS = frozenset(
    {
        torch.ops._c10d_functional.all_reduce,
        torch.ops._c10d_functional.all_gather_into_tensor,
        torch.ops._c10d_functional.reduce_scatter_tensor,
        torch.ops._c10d_functional.all_to_all_single,
        torch.ops._c10d_functional.wait_tensor,
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


def _is_collective_or_wait(node: Node) -> bool:
    """Check if a node is a distributed collective or wait op."""
    target = node.target
    if hasattr(target, "overloadpacket"):
        return target.overloadpacket in _COLLECTIVE_OPS
    return target in _COLLECTIVE_OPS


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


# ============================================================
# Forward/backward node classification for make_fx traced graphs
# ============================================================


def _classify_forward_backward(
    gm: torch.fx.GraphModule,
) -> tuple[set[Node], set[Node]]:
    """Classify nodes in a make_fx traced joint graph as forward or backward.

    Uses ``autograd_backward`` metadata set by PyTorch's autograd engine during
    make_fx tracing. This is the same classification used by the SAC remat pass,
    ensuring consistent forward/backward boundaries across graph passes.

    Nodes without ``autograd_backward`` metadata (placeholders, output, newly
    inserted nodes) are excluded from both sets.

    Returns:
        Tuple of (forward_nodes, backward_nodes).
    """
    forward_nodes: set[Node] = set()
    backward_nodes: set[Node] = set()

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "get_attr"):
            continue
        if node.meta.get("autograd_backward", False):
            backward_nodes.add(node)
        else:
            forward_nodes.add(node)

    return forward_nodes, backward_nodes


# ============================================================
# Tagging pass
# ============================================================


def tag_all_offloadable_activations(gm: torch.fx.GraphModule) -> None:
    """Tag all saved activations eligible for CPU offloading.

    This is a reference implementation that offloads all offloadable
    activations. In practice, users should tag activations via annotations
    or graph passes for more fine-grained control over which activations
    are offloaded.

    Sets ``node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD`` on
    eligible nodes. Eligibility requires:
      1. Node is a forward node (determined via seq_nr)
      2. Node has ``meta["custom"]["ac_region_id"]`` (layer boundary marker)
      3. Passes ``_can_offload_node`` (not a view, not tiny, etc.)
      4. Has at least one backward consumer

    The last layer is skipped when multiple layers exist, since its activations
    are consumed immediately in backward (offloading adds overhead with no
    benefit).

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
    """
    forward_nodes, backward_nodes = _classify_forward_backward(gm)

    # Find all layer IDs to determine the last layer
    all_layer_ids: set[int] = set()
    for node in gm.graph.nodes:
        lid = node.meta.get("custom", {}).get(_AC_REGION_ID)
        if lid is not None:
            all_layer_ids.add(lid)
    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else None

    tagged = 0
    total_bytes = 0

    for node in gm.graph.nodes:
        if node not in forward_nodes:
            continue
        layer_id = node.meta.get("custom", {}).get(_AC_REGION_ID)
        # Skip the last layer: its activations are consumed immediately in
        # backward, so offloading adds overhead with no memory benefit.
        if layer_id is None or layer_id == last_layer_id:
            continue
        if not _can_offload_node(node):
            continue

        # Check for backward consumers
        has_backward_users = any(u in backward_nodes for u in node.users)
        if not has_backward_users:
            continue

        node.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
        tagged += 1
        val = node.meta.get("val")
        if val is not None:
            total_bytes += _tensor_bytes(val)

    logger.info(
        f"CPU offload: tagged {tagged} activations for offload "
        f"({total_bytes / 1024 / 1024:.2f} MB)"
    )


# ============================================================
# Graph passes
# ============================================================


def apply_cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Insert ao offload/reload/wait_tensor ops for nodes tagged ``MUST_CPU_OFFLOAD``.

    Reads ``node.meta["recompute"] is CheckpointPolicy.MUST_CPU_OFFLOAD`` (set by
    ``tag_all_offloadable_activations``) and inserts:
      Forward:  cpu = ao.offload(gpu_tensor)
                cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor)
      Backward: gpu = ao.reload(cpu, device)
                gpu = ao.wait_tensor(gpu)
    Then redirects backward consumers to use the reloaded tensor.

    The reload device is inferred from the original tensor's device
    (from ``node.meta["val"].device``) before offloading.

    Uses ``meta["custom"]["ac_region_id"]`` and seq_nr-based forward/backward
    classification to determine layer boundaries.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by graph pass interface).

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    # 1. Collect nodes tagged for offload, with their backward consumers
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

    # 2. Build position index for ordering queries (only used for pre-existing
    # nodes; newly inserted nodes never appear in bwd_users).
    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # 3. Insert offload/reload/wait_tensor ops
    total_bytes = 0

    for node, bwd_users in offloadable:
        val = node.meta.get("val")
        assert val is not None, (
            f"Node {node.name} tagged for offload has no 'val' metadata"
        )

        # Infer reload device from the original tensor's device before offload
        device = val.device

        # --- Forward: async GPU->CPU offload right after production ---
        with gm.graph.inserting_after(node):
            offload_node = gm.graph.call_function(
                torch.ops.ao.offload.default,
                args=(node,),
            )
            offload_node.meta["val"] = val.to(torch.device("cpu"))

        # --- Forward: wait_tensor for D2H completion (initially after offload) ---
        with gm.graph.inserting_after(offload_node):
            wait_offload_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(offload_node, node),
            )
            wait_offload_node.meta["val"] = offload_node.meta["val"]

        # --- Backward: async CPU->GPU reload + wait before first consumer ---
        first_consumer = min(bwd_users, key=lambda n: node_to_index[n])

        with gm.graph.inserting_before(first_consumer):
            reload_node = gm.graph.call_function(
                torch.ops.ao.reload.default,
                args=(wait_offload_node, device),
            )
            reload_node.meta["val"] = val
            reload_node.meta["autograd_backward"] = True

        with gm.graph.inserting_before(first_consumer):
            wait_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(reload_node,),
            )
            wait_node.meta["val"] = val
            wait_node.meta["autograd_backward"] = True

        # Redirect backward consumers to the reloaded tensor
        for user in bwd_users:
            user.replace_input_with(node, wait_node)

        logger.debug(
            f"CPU offload: offloading {node.name} "
            f"({_tensor_bytes(val) / 1024:.1f} KB, {val.shape})"
        )
        total_bytes += _tensor_bytes(val)

    # TODO: Add scheduling optimizations (defer offload waits, prefetch reloads)
    # for D2H/H2D overlap with compute. The make_fx traced joint graph
    # interleaves forward and backward nodes (unlike manually built graphs
    # where they are cleanly separated), so the simple "defer to next layer's
    # forward" / "prefetch to preceding layer's backward" strategy from the
    # reference implementation needs adaptation. For now, offload/reload ops
    # are placed right next to their producers/consumers, which still saves
    # memory but without async transfer overlap.

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"CPU offload: offloaded {len(offloadable)} tensors "
        f"({total_bytes / 1024 / 1024:.2f} MB)"
    )
    return gm


def cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Two-phase CPU offload pass: tag eligible activations, then insert ops.

    This is the top-level entry point conforming to the graph pass signature.
    It first tags eligible activations with ``tag_all_offloadable_activations``,
    then inserts offload/reload/wait ops with ``apply_cpu_offload_pass``.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by pass interface).

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    tag_all_offloadable_activations(gm)
    return apply_cpu_offload_pass(gm, example_inputs)
