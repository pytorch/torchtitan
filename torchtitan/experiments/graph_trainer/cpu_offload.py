# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU offload pass for activation offloading in graph_trainer.

Defines custom ops (ao:: namespace) for async GPU<->CPU transfers with
event-based synchronization, and graph passes that insert them around
saved activations to reduce GPU memory.

Offload pattern (forward):
    cpu = ao.offload(gpu_tensor)
    ... forward consumers use gpu_tensor ...
    cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor)

Reload pattern (backward):
    gpu = ao.reload(cpu, device)
    gpu = ao.wait_tensor(gpu)
    ... backward consumers use gpu ...

This module works with the make_fx traced joint fwd+bwd graph, using
``meta["custom"]["ac_region_id"]`` as layer boundaries and ``seq_nr``
to distinguish forward from backward nodes.
"""

import enum
import operator

import torch
import torch.fx
from torch._library.custom_ops import custom_op
from torch._logging import trace_structured
from torch.fx import has_side_effect, Node

from torchtitan.experiments.graph_trainer.common_utils import _AC_REGION_ID
from torchtitan.tools.logging import logger

aten = torch.ops.aten


# ============================================================
# Custom ops for async activation offloading (ao:: namespace)
#
# A single dedicated transfer stream handles all D2H/H2D copies.
# Completion events are keyed by output tensor data_ptr() and
# cached for reuse across CUDA graph replays.
#
# ``keepalive`` in the offload wait_tensor creates a graph dependency
# that extends the GPU tensor's lifetime past the async D2H copy,
# replacing ``record_stream`` (which causes memory fragmentation).
# ============================================================

# --- Stream and transfer registry (lazily created, cached) ---
_transfer_stream: torch.cuda.Stream | None = None

# Maps output tensor data_ptr() -> (completion_event, device_to_sync_on).
# Created lazily by _record_wait, cached for reuse across CUDA graph replays.
_pending_transfers: dict[int, tuple[torch.cuda.Event, torch.device]] = {}


def _get_transfer_stream(device: torch.device) -> torch.cuda.Stream:
    """Get or create the dedicated transfer stream."""
    global _transfer_stream
    if _transfer_stream is None:
        _transfer_stream = torch.cuda.Stream(device)
    return _transfer_stream


def _record_wait(tensor: torch.Tensor, device: torch.device) -> torch.cuda.Event:
    """Create an event for an async transfer and register it."""
    key = tensor.data_ptr()
    event = torch.cuda.Event()
    _pending_transfers[key] = (event, device)
    return event


def _pop_wait(tensor: torch.Tensor) -> tuple[torch.cuda.Event, torch.device]:
    """Pop the (event, device) registered by _record_wait for this tensor."""
    key = tensor.data_ptr()
    try:
        return _pending_transfers.pop(key)
    except KeyError:
        raise RuntimeError(
            f"ao.wait_tensor: no pending transfer for tensor with data_ptr={key}. "
            "Every ao.wait_tensor must be paired with a preceding ao.offload or ao.reload."
        ) from None


@custom_op("ao::offload", mutates_args=())
def offload(tensor: torch.Tensor) -> torch.Tensor:
    """Async offload a GPU tensor to CPU on the dedicated transfer stream.

    Callers MUST pair this with an ``ao.wait_tensor`` that passes the source
    GPU tensor as ``keepalive`` to extend its lifetime past the async D2H copy.
    """
    device = tensor.device
    transfer_stream = _get_transfer_stream(device)
    current_stream = torch.cuda.current_stream(device)

    transfer_stream.wait_stream(current_stream)

    torch.cuda.set_stream(transfer_stream)
    result = torch.empty_like(tensor, device="cpu", pin_memory=True)
    completion_event = _record_wait(result, device)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.cuda.set_stream(current_stream)

    return result


@offload.register_fake
def _offload_fake(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu")


@custom_op("ao::reload", mutates_args=())
def reload(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on the dedicated transfer stream.

    The GPU tensor is allocated on the compute stream to avoid cross-stream
    allocator ownership issues. The H2D copy runs on the transfer stream.
    """
    transfer_stream = _get_transfer_stream(device)
    current_stream = torch.cuda.current_stream(device)

    # Allocate on compute stream so the allocator tracks ownership correctly
    result = torch.empty_like(tensor, device=device)
    completion_event = _record_wait(result, device)

    # Ensure transfer stream sees the allocation before starting the copy.
    # When CUDAGraph is enabled, this wait_stream ensures the reload happens
    # at the correct time.
    transfer_stream.wait_stream(current_stream)

    torch.cuda.set_stream(transfer_stream)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.cuda.set_stream(current_stream)

    return result


@reload.register_fake
def _reload_fake(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_like(tensor, device=device)


# ---------------------------------------------------------------------------
# ao::wait_tensor: defined via torch.library with an aliasing schema so the
# output can alias the input (custom_op forbids output aliasing input).
#
# Uses CompositeExplicitAutograd (single impl for all devices) because the
# offload case has mixed-device args: ``tensor`` is CPU (the offload result)
# while ``keepalive`` is CUDA (the source GPU tensor).
# ---------------------------------------------------------------------------
_lib = torch.library.Library("ao", "DEF")
_lib.define("wait_tensor(Tensor(a) tensor, Tensor? keepalive=None) -> Tensor(a)")


@torch.library.impl("ao::wait_tensor", "CompositeExplicitAutograd")
def _ao_wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
) -> torch.Tensor:
    completion_event, device = _pop_wait(tensor)
    current_stream = torch.cuda.current_stream(device)
    current_stream.wait_event(completion_event)
    return tensor


@torch.library.register_fake("ao::wait_tensor")
def _ao_wait_tensor_fake(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.wait_tensor.default)


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


class OffloadPolicy(enum.Enum):
    """Recompute policy for CPU offloading, stored in node.meta["recompute"]."""

    MUST_OFFLOAD = "must_offload"


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


def _can_offload_node(node: Node) -> bool:
    """Check if a node's output can be profitably offloaded to CPU.

    Excludes views (offloading doesn't free base tensor memory), small tensors
    (overhead exceeds benefit), non-contiguous tensors, and collective/wait ops.

    getitem nodes are allowed -- they unpack multi-output ops and produce
    real tensors, not views.
    """
    if node.op != "call_function":
        return False
    # getitem unpacks tuples from multi-output ops; the result tensor has its
    # own allocation. Allow offloading.
    if node.target is operator.getitem:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return False
        if not _tensor_is_contiguous(val):
            return False
        if _tensor_bytes(val) < _MIN_OFFLOAD_BYTES:
            return False
        return True
    if _is_view(node):
        return False
    if _is_collective_or_wait(node):
        return False
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return False
    if not _tensor_is_contiguous(val):
        return False
    if _tensor_bytes(val) < _MIN_OFFLOAD_BYTES:
        return False
    return True


# ============================================================
# Forward/backward node classification for make_fx traced graphs
# ============================================================


def _classify_forward_backward(
    gm: torch.fx.GraphModule,
) -> tuple[set[Node], set[Node]]:
    """Classify nodes in a make_fx traced joint graph as forward or backward.

    Uses ``seq_nr`` metadata: for each unique seq_nr, the first node seen is
    treated as forward, and subsequent nodes with the same seq_nr are backward.
    Nodes without seq_nr (placeholders, output, etc.) are excluded from both sets.

    Returns:
        Tuple of (forward_nodes, backward_nodes).
    """
    seq_nr_to_first: dict[int, Node] = {}
    forward_nodes: set[Node] = set()
    backward_nodes: set[Node] = set()

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "get_attr"):
            continue
        seq_nr = node.meta.get("seq_nr")
        if seq_nr is None:
            continue
        if seq_nr not in seq_nr_to_first:
            seq_nr_to_first[seq_nr] = node
            forward_nodes.add(node)
        else:
            backward_nodes.add(node)

    return forward_nodes, backward_nodes


# ============================================================
# Tagging pass
# ============================================================


def tag_offloadable_activations(gm: torch.fx.GraphModule) -> None:
    """Tag saved activations eligible for CPU offloading.

    Sets ``node.meta["recompute"] = OffloadPolicy.MUST_OFFLOAD`` on eligible
    nodes. Eligibility requires:
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
        if layer_id is None or layer_id == last_layer_id:
            continue
        if not _can_offload_node(node):
            continue

        # Check for backward consumers
        has_backward_users = any(u in backward_nodes for u in node.users)
        if not has_backward_users:
            continue

        node.meta["recompute"] = OffloadPolicy.MUST_OFFLOAD
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


def _tlparse_log_graph(gm: torch.fx.GraphModule, graph_name: str) -> None:
    """Log the graph to tlparse via trace_structured."""
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": graph_name,
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        ),
        expect_trace_id=False,
    )


def _find_fwd_layer_end_nodes(
    gm: torch.fx.GraphModule,
    forward_nodes: set[Node],
) -> dict[int, Node]:
    """Find the last forward node per layer (ac_region_id)."""
    fwd_end: dict[int, Node] = {}
    for node in gm.graph.nodes:
        if node not in forward_nodes:
            continue
        layer_id = node.meta.get("custom", {}).get(_AC_REGION_ID)
        if layer_id is None:
            continue
        fwd_end[layer_id] = node  # keeps getting overwritten; last one wins
    return fwd_end


def _find_bwd_layer_start_nodes(
    gm: torch.fx.GraphModule,
    backward_nodes: set[Node],
) -> dict[int, Node]:
    """Find the first backward node per layer (ac_region_id)."""
    bwd_start: dict[int, Node] = {}
    for node in gm.graph.nodes:
        if node not in backward_nodes:
            continue
        layer_id = node.meta.get("custom", {}).get(_AC_REGION_ID)
        if layer_id is None:
            continue
        if layer_id not in bwd_start:
            bwd_start[layer_id] = node
    return bwd_start


def _defer_offload_waits(
    gm: torch.fx.GraphModule,
    offload_wait_info: list[tuple[Node, int]],
    fwd_end_node_map: dict[int, Node],
) -> None:
    """Move offload wait_tensor ops to end of the next layer's forward pass.

    For each offload wait belonging to layer N, moves it to just after the
    last forward node of layer N+1. The ao.offload stays at its original
    position (right after the tensor's production), so there is a large gap
    between offload and wait where the async D2H transfer can overlap with
    forward compute.

    Falls back to the current layer's end if no next layer exists.
    """
    if not offload_wait_info or not fwd_end_node_map:
        return

    # Build next-layer mapping from sorted layer IDs
    sorted_layers = sorted(fwd_end_node_map.keys())
    layer_to_next: dict[int, int] = {}
    for i, lid in enumerate(sorted_layers):
        if i + 1 < len(sorted_layers):
            layer_to_next[lid] = sorted_layers[i + 1]
        else:
            layer_to_next[lid] = lid  # last layer falls back to itself

    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}
    moved = 0

    for wait_node, layer_idx in offload_wait_info:
        target_layer = layer_to_next.get(layer_idx, layer_idx)
        target = fwd_end_node_map.get(target_layer)
        if target is None:
            continue

        # Check dependencies: wait must come after all its inputs
        dep_pos = max(node_to_index.get(inp, 0) for inp in wait_node.all_input_nodes)
        target_pos = node_to_index.get(target, 0)

        if target_pos <= dep_pos:
            continue  # Can't move past dependencies

        cur_pos = node_to_index.get(wait_node, 0)
        if cur_pos >= target_pos:
            continue  # Already at or past target

        # Move wait to just after the last forward node of the next layer
        target.append(wait_node)
        moved += 1

    if moved:
        logger.info(
            f"CPU offload defer: moved {moved} offload wait ops to end of next layer"
        )


def _prefetch_reloads(
    gm: torch.fx.GraphModule,
    reload_info: list[tuple[Node, int]],
    bwd_start_node_map: dict[int, Node],
) -> None:
    """Move reload ops to preceding layer's backward for H2D/compute overlap.

    For each reload belonging to layer N, moves it to the start of layer
    (N+1)'s backward (which executes before layer N's backward since the
    backward pass goes from layer N-1 down to 0). The corresponding
    ao.wait_tensor stays at its original position, so there is a gap
    between reload and wait where the async H2D transfer can overlap with
    backward compute.

    For the highest layer (no preceding backward), reloads stay in place.
    """
    if not reload_info or not bwd_start_node_map:
        return

    # Backward layers execute in reverse order: N-1, N-2, ..., 0
    bwd_layer_order = sorted(bwd_start_node_map.keys(), reverse=True)
    if not bwd_layer_order:
        return

    # Map each layer to its prefetch target layer (the one whose backward
    # runs just before it, i.e. the layer with the next-higher index)
    layer_to_prefetch: dict[int, int | None] = {}
    for i, layer in enumerate(bwd_layer_order):
        if i == 0:
            layer_to_prefetch[layer] = None  # highest layer
        else:
            layer_to_prefetch[layer] = bwd_layer_order[i - 1]

    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}
    moved = 0

    for reload_node, layer_idx in reload_info:
        prefetch_layer = layer_to_prefetch.get(layer_idx)
        if prefetch_layer is None:
            continue  # Highest layer: can't prefetch, keep in place

        target = bwd_start_node_map.get(prefetch_layer)
        if target is None:
            continue

        # Check dependencies: reload must come after all its inputs
        dep_pos = max(node_to_index.get(inp, 0) for inp in reload_node.all_input_nodes)
        target_pos = node_to_index.get(target, 0)

        if target_pos <= dep_pos:
            continue  # Can't move before dependencies

        cur_pos = node_to_index.get(reload_node, 0)
        if cur_pos <= target_pos:
            continue  # Already before target

        # Move reload to just before the target layer's backward start
        target.prepend(reload_node)
        moved += 1

    if moved:
        logger.info(f"CPU offload prefetch: moved {moved} reload ops")


def _apply_cpu_offload_pass(
    gm: torch.fx.GraphModule,
    device: torch.device,
) -> torch.fx.GraphModule:
    """Insert ao offload/reload/wait_tensor ops for nodes tagged ``MUST_OFFLOAD``.

    Reads ``node.meta["recompute"] is OffloadPolicy.MUST_OFFLOAD`` (set by
    ``tag_offloadable_activations``) and inserts:
      Forward:  cpu = ao.offload(gpu_tensor)
                cpu = ao.wait_tensor(cpu, keepalive=gpu_tensor)
      Backward: gpu = ao.reload(cpu, device)
                gpu = ao.wait_tensor(gpu)
    Then redirects backward consumers to use the reloaded tensor.

    Uses ``meta["custom"]["ac_region_id"]`` and seq_nr-based forward/backward
    classification to determine layer boundaries for prefetching.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        device: The GPU device for reload targets.
    """
    _tlparse_log_graph(gm, graph_name="cpu_offload_before")

    forward_nodes, backward_nodes = _classify_forward_backward(gm)

    # 1. Collect nodes tagged for offload, with their backward consumers
    offloadable: list[tuple[Node, list[Node], int]] = []
    for node in gm.graph.nodes:
        if node.meta.get("recompute") is not OffloadPolicy.MUST_OFFLOAD:
            continue
        bwd_users = [u for u in node.users if u in backward_nodes]
        if not bwd_users:
            continue
        layer_idx = node.meta.get("custom", {}).get(_AC_REGION_ID, 0)
        offloadable.append((node, bwd_users, layer_idx))

    if not offloadable:
        return gm

    # 2. Build position index for ordering queries
    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # 3. Insert offload/reload/wait_tensor ops
    total_bytes = 0
    reload_info: list[tuple[Node, int]] = []  # for prefetch pass
    offload_wait_info: list[tuple[Node, int]] = []  # for defer pass

    for node, bwd_users, layer_idx in offloadable:
        val = node.meta.get("val")
        if val is None:
            continue

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

        offload_wait_info.append((wait_offload_node, layer_idx))

        # --- Backward: async CPU->GPU reload + wait before first consumer ---
        first_consumer = min(bwd_users, key=lambda n: node_to_index[n])

        with gm.graph.inserting_before(first_consumer):
            reload_node = gm.graph.call_function(
                torch.ops.ao.reload.default,
                args=(wait_offload_node, device),
            )
            reload_node.meta["val"] = val

        with gm.graph.inserting_before(first_consumer):
            wait_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(reload_node,),
            )
            wait_node.meta["val"] = val

        # Redirect backward consumers to the reloaded tensor
        for user in bwd_users:
            user.replace_input_with(node, wait_node)

        reload_info.append((reload_node, layer_idx))
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
    _tlparse_log_graph(gm, graph_name="cpu_offload_after")
    logger.info(
        f"CPU offload: offloaded {len(offloadable)} tensors "
        f"({total_bytes / 1024 / 1024:.2f} MB)"
    )
    return gm


def cpu_offload_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    device: torch.device | None = None,
) -> torch.fx.GraphModule:
    """Two-phase CPU offload pass: tag eligible activations, then insert ops.

    This is the top-level entry point conforming to the graph pass signature.
    It first tags eligible activations with ``tag_offloadable_activations``,
    then inserts offload/reload/wait ops with ``_apply_cpu_offload_pass``.

    If no device is provided, attempts to infer it from the graph's placeholder
    metadata. Falls back to ``torch.device("cuda")`` if inference fails.

    Args:
        gm: The GraphModule containing the full fwd+bwd graph.
        example_inputs: Example inputs (unused, required by pass interface).
        device: The GPU device for reload targets. Auto-detected if None.

    Returns:
        The transformed GraphModule with offload/reload ops inserted.
    """
    if device is None:
        # Infer device from the first tensor placeholder
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor) and val.device.type == "cuda":
                    device = val.device
                    break
        if device is None:
            device = torch.device("cuda")

    tag_offloadable_activations(gm)
    return _apply_cpu_offload_pass(gm, device)
