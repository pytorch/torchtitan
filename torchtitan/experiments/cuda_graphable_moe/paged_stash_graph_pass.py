# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Paged stash graph pass for aot_fx_trace mode.

apply_paged_stash_pass: Identifies stash-eligible MoE fc1 activations in the
joint fwd+bwd graph via op-target + consumer-pattern matching, inserts
``paged_stash.copy`` + ``ao.wait_tensor`` after the forward producer, and
``paged_stash.pop`` + ``ao.wait_tensor`` before the backward consumers.
Backward consumers are redirected to read from the pop output instead of the
original activation.  After this pass, SAC's remat sees only the compact
``page_record`` (int64 handle) crossing the fwd→bwd boundary — the large
activation has no backward users and is freed after forward.

Stash-eligible nodes are identified by matching ``aten._grouped_mm.default``
nodes whose output feeds ``aten.silu`` (gate projection) or ``aten.mul``
(up projection).  The down projection (fc2) feeds ``aten.to.dtype`` and is
excluded.
"""

import operator

import torch

# Side-effect import: registers ao::wait_tensor custom op
import torch._functorch._activation_offloading.offload_ops as offload_ops  # noqa: F401
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_checkpoint import _get_save_ops
from torchtitan.experiments.graph_trainer.passes import apply_sac_pass

from torchtitan.tools.logging import logger

# Import to ensure paged_stash::copy/pop custom ops are registered
from . import paged_stash_ops  # noqa: F401
from .paged_stash_ops import PagedStashBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_moe_fc1_grouped_mm(node: fx.Node) -> bool:
    """Check if a node is a MoE fc1 (gate or up) ``_grouped_mm`` output.

    Matches ``aten._grouped_mm.default`` nodes that are part of the SwiGLU
    fc1 computation:
    - **fc1_1 (gate)**: output feeds ``aten.silu`` directly.
    - **fc1_2 (up)**: output feeds ``aten.mul`` where the *other* operand
      comes from ``aten.silu`` (i.e., the gated activation ``silu(fc1_1) * fc1_2``).

    The fc2 down projection also feeds ``aten.mul`` (combine with shared
    experts), but its ``mul`` partner is not a ``silu`` output — so it is
    excluded.
    """
    if node.op != "call_function":
        return False
    if node.target != torch.ops.aten._grouped_mm.default:
        return False
    for user in node.users:
        if user.op != "call_function":
            continue
        if user.target == torch.ops.aten.silu.default:
            return True
        if user.target == torch.ops.aten.mul.Tensor:
            for mul_input in user.all_input_nodes:
                if mul_input is not node and _is_silu_output(mul_input):
                    return True
    return False


def _is_silu_output(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target == torch.ops.aten.silu.default


def _has_dynamic_first_dim(node: fx.Node) -> bool:
    """Check if a node's first dimension is dynamic (SymInt)."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return False
    return isinstance(val.shape[0], torch.SymInt)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: find actual token count for _grouped_mm outputs
# ---------------------------------------------------------------------------


def _find_num_tokens_node(
    gm: fx.GraphModule,
    fwd_node: fx.Node,
    insert_before: fx.Node,
    *,
    _cache: dict[int, fx.Node] | None = None,
) -> fx.Node | None:
    """Find a graph node representing the actual token count for a saved tensor.

    For ``_grouped_mm`` outputs, the offsets tensor (from ``cumsum``) is
    available via ``kwargs["offs"]``.  ``offsets[-1]`` equals
    ``tokens_per_expert.sum()`` — the actual (non-padded) token count.

    For derived tensors (transposes, casts, etc.), walks backward through
    the producer chain to find the originating ``_grouped_mm`` node.

    Returns an int64 scalar node, or None if not found.
    """
    if _cache is None:
        _cache = {}

    if (
        fwd_node.op == "call_function"
        and fwd_node.target == torch.ops.aten._grouped_mm.default
    ):
        offsets_node = fwd_node.kwargs.get("offs")
        if offsets_node is None and len(fwd_node.args) > 2:
            offsets_node = fwd_node.args[2]
        if offsets_node is None or not isinstance(offsets_node, fx.Node):
            return None

        cache_key = id(offsets_node)
        if cache_key in _cache:
            return _cache[cache_key]

        with gm.graph.inserting_before(insert_before):
            total_node = gm.graph.call_function(
                torch.ops.aten.select.int,
                args=(offsets_node, 0, -1),
            )
            offsets_val = offsets_node.meta.get("val")
            if offsets_val is not None:
                total_node.meta["val"] = offsets_val.select(0, -1)
            total_i64 = gm.graph.call_function(
                torch.ops.aten.to.dtype,
                args=(total_node, torch.int64),
            )
            if "val" in total_node.meta:
                total_i64.meta["val"] = total_node.meta["val"].to(torch.int64)
        _cache[cache_key] = total_i64
        return total_i64

    for arg in fwd_node.all_input_nodes:
        result = _find_num_tokens_node(gm, arg, insert_before, _cache=_cache)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Joint-graph pass: fwd/bwd classification
# ---------------------------------------------------------------------------


def _classify_forward_backward(gm: fx.GraphModule):
    """Classify nodes as forward or backward.

    Uses ``autograd_backward`` metadata set by the make_fx tracer in
    ``aot_fx_trace`` mode.
    """
    forward_nodes: set[fx.Node] = set()
    backward_nodes: set[fx.Node] = set()

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.meta.get("autograd_backward", False):
            backward_nodes.add(node)
        else:
            forward_nodes.add(node)

    return forward_nodes, backward_nodes


# ---------------------------------------------------------------------------
# Joint-graph pass: eligibility check
# ---------------------------------------------------------------------------


def _is_paged_stash_eligible(
    node: fx.Node,
    forward_nodes: set[fx.Node],
    backward_nodes: set[fx.Node],
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> tuple[list[fx.Node], PagedStashBuffer] | None:
    """Check if a forward node is eligible for paged stash.

    Returns ``(bwd_consumers, buffer)`` if eligible, or ``None``.

    A node is eligible when all of:
    1. It is a forward ``call_function`` node.
    2. It is an ``aten._grouped_mm`` whose consumer is ``silu`` or ``mul``
       (MoE fc1 gate/up projection, identified by op-target + consumer pattern).
    3. It has real (non-sym) backward consumers.
    4. It has a dynamic SymInt first dimension.
    5. Its ``(dtype, shape[-1])`` matches a pre-allocated paged buffer.
    """
    if node.op != "call_function":
        return None
    if node not in forward_nodes:
        return None
    if not _is_moe_fc1_grouped_mm(node):
        return None

    bwd_consumers = [u for u in node.users if u in backward_nodes]
    if not bwd_consumers:
        return None
    if all(is_sym_node(u) for u in bwd_consumers):
        return None
    if not _has_dynamic_first_dim(node):
        return None

    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return None
    key = (val.dtype, val.shape[-1])
    buf = paged_buffers.get(key)
    if buf is None:
        return None

    return bwd_consumers, buf


# ---------------------------------------------------------------------------
# Joint-graph pass: main entry point
# ---------------------------------------------------------------------------


def apply_paged_stash_pass(
    gm: torch.fx.GraphModule,
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> torch.fx.GraphModule:
    """Insert paged_stash.copy/pop + ao.wait_tensor into the joint graph.

    1. Classify forward vs backward nodes using ``autograd_backward``.
    2. For each eligible forward node (fc1 _grouped_mm, dynamic, has bwd consumers):
       a. Insert ``paged_stash.copy`` + ``ao.wait_tensor`` after the fwd node.
       b. Insert ``paged_stash.pop`` + ``ao.wait_tensor`` before bwd consumers.
       c. Redirect backward consumers to read from the pop output.
    3. After this pass, SAC's remat sees ``page_record`` (small int64) crossing
       the boundary, not the large activation.

    Args:
        gm: The joint forward-backward graph module.
        paged_buffers: Dict mapping ``(dtype, hidden_size)`` to
            ``PagedStashBuffer``.

    Returns:
        The modified graph module.
    """
    from .paged_stash_ops import register_paged_stash_buffer

    forward_nodes, backward_nodes = _classify_forward_backward(gm)

    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    # Ensure all buffers are registered in the module-level registry.
    # register_paged_stash_buffer is idempotent for the same buffer object.
    buffer_ids: dict[int, int] = {}
    for buf in paged_buffers.values():
        buf_key = id(buf)
        if buf_key not in buffer_ids:
            buffer_ids[buf_key] = register_paged_stash_buffer(buf)

    # Get FakeTensorMode from the joint graph for creating proper metadata
    # on inserted nodes (the partitioner requires FakeTensor, not meta tensors).
    fake_mode = None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            v = node.meta["val"]
            if hasattr(v, "fake_mode"):
                fake_mode = v.fake_mode
                break

    def _make_fake(shape, dtype=torch.int64):
        """Create a FakeTensor with the correct mode for the partitioner."""
        if fake_mode is not None:
            with fake_mode:
                return torch.empty(shape, dtype=dtype, device="cuda")
        return torch.empty(shape, dtype=dtype, device="meta")

    # Collect eligible nodes in topological order
    eligible: list[tuple[fx.Node, list[fx.Node], PagedStashBuffer]] = []
    for node in gm.graph.nodes:
        result = _is_paged_stash_eligible(
            node, forward_nodes, backward_nodes, paged_buffers
        )
        if result is not None:
            bwd_consumers, buf = result
            eligible.append((node, bwd_consumers, buf))
            # SAC must have already annotated these nodes. They should be
            # PREFER_RECOMPUTE (expert activations are not in SAC's save
            # list), meaning SAC would recompute them — paged stash saves
            # them via paging instead. If SAC marked them MUST_SAVE, the
            # paged stash is redundant; if SAC hasn't run, something is
            # wrong with pass ordering.
            sac_policy = node.meta.get("recompute")
            if sac_policy is None:
                if _rank0:
                    logger.debug(
                        "Paged stash node %s has no SAC annotation "
                        "(SAC may not have run yet).",
                        node.name,
                    )
            elif _rank0 and sac_policy != CheckpointPolicy.PREFER_RECOMPUTE:
                logger.warning(
                    "Paged stash node %s has SAC policy %s (expected "
                    "PREFER_RECOMPUTE). Paged stash will override.",
                    node.name,
                    sac_policy,
                )

    if not eligible:
        logger.info("apply_paged_stash_pass: no eligible nodes found")
        return gm

    node_to_index = {n: i for i, n in enumerate(gm.graph.nodes)}
    num_tokens_cache: dict[int, fx.Node] = {}

    for fwd_node, bwd_consumers, buf in eligible:
        buffer_id = buffer_ids[id(buf)]
        val = fwd_node.meta["val"]

        # --- Forward: insert copy + wait_tensor after fwd_node ---

        first_bwd_consumer = min(bwd_consumers, key=lambda n: node_to_index[n])

        # Find actual token count (offsets[-1] from _grouped_mm kwargs).
        # Helper nodes (select, to.dtype) are inserted right after fwd_node.
        num_tokens_node = _find_num_tokens_node(
            gm, fwd_node, fwd_node.next, _cache=num_tokens_cache
        )
        if _rank0:
            found_str = (
                "offsets[-1]"
                if num_tokens_node is not None
                else "tensor shape fallback"
            )
            logger.debug("  num_tokens for %s: %s", fwd_node.name, found_str)

        # Walk past any helper nodes that _find_num_tokens_node just created
        # so that our copy node is inserted after them (topological order).
        insert_pt = fwd_node
        cursor = fwd_node.next
        while (
            cursor is not None
            and cursor.op == "call_function"
            and cursor.target
            in (
                torch.ops.aten.select.int,
                torch.ops.aten.to.dtype,
            )
        ):
            insert_pt = cursor
            cursor = cursor.next

        if num_tokens_node is not None:
            actual_num_tokens = num_tokens_node
        else:
            with gm.graph.inserting_after(fwd_node):
                num_tokens_fallback = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=([1], val.shape[0]),
                    kwargs={"dtype": torch.int64, "device": val.device},
                )
                num_tokens_fallback.meta["val"] = _make_fake(1, torch.int64)
            insert_pt = num_tokens_fallback
            actual_num_tokens = num_tokens_fallback

        # Compute fake metadata for the inserted nodes
        flat_shape = val.reshape(-1, buf.hidden_size)
        max_num_tokens = flat_shape.shape[0]
        max_num_pages = (max_num_tokens + buf.page_size - 1) // buf.page_size

        page_record_fake = _make_fake(max_num_pages + 2, torch.int64)
        new_head_fake = _make_fake(2, torch.int64)
        pop_fake = _make_fake((max_num_tokens, buf.hidden_size), val.dtype)

        with gm.graph.inserting_after(insert_pt):
            copy_node = gm.graph.call_function(
                torch.ops.paged_stash.copy,
                args=(
                    fwd_node,
                    buf.page_size,
                    buf.hidden_size,
                    actual_num_tokens,
                    buffer_id,
                ),
            )
            copy_node.meta["val"] = (page_record_fake, new_head_fake)
            # MUST_SAVE so the partitioner doesn't treat it as impure
            copy_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            copy_node.meta["ac_graph_id"] = 0
            copy_node.meta["autograd_backward"] = False

        with gm.graph.inserting_after(copy_node):
            page_record_node = gm.graph.call_function(
                operator.getitem,
                args=(copy_node, 0),
            )
            page_record_node.meta["val"] = page_record_fake
            page_record_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            page_record_node.meta["ac_graph_id"] = 0
            page_record_node.meta["autograd_backward"] = False

        with gm.graph.inserting_after(page_record_node):
            # keepalive=fwd_node extends the activation's lifetime past the
            # async Triton copy on the transfer stream (same pattern as
            # ao.wait_tensor(offload_result, gpu_tensor) in cpu_offload_pass)
            wait_copy_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(page_record_node, fwd_node),
            )
            wait_copy_node.meta["val"] = page_record_fake
            # ao.wait_tensor has has_side_effect, which the partitioner treats
            # as impure. MUST_SAVE bypasses the impure-op assertion.
            wait_copy_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            wait_copy_node.meta["ac_graph_id"] = 0
            wait_copy_node.meta["autograd_backward"] = False

        # --- Backward: insert pop + wait_tensor before first bwd consumer ---

        with gm.graph.inserting_before(first_bwd_consumer):
            pop_node = gm.graph.call_function(
                torch.ops.paged_stash.pop,
                args=(
                    wait_copy_node,
                    buf.page_size,
                    buf.hidden_size,
                    val.dtype,
                    buffer_id,
                ),
            )
            pop_node.meta["val"] = pop_fake
            pop_node.meta["autograd_backward"] = True

        with gm.graph.inserting_after(pop_node):
            wait_pop_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(pop_node,),
            )
            wait_pop_node.meta["val"] = pop_fake
            wait_pop_node.meta["autograd_backward"] = True

        if len(val.shape) > 2:
            with gm.graph.inserting_after(wait_pop_node):
                restore_node = gm.graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(wait_pop_node, list(val.shape)),
                )
                restore_node.meta["val"] = val
                restore_node.meta["autograd_backward"] = True
        else:
            restore_node = wait_pop_node

        # Redirect backward consumers from fwd_node to the restored tensor
        for bwd_user in bwd_consumers:
            bwd_user.replace_input_with(fwd_node, restore_node)

    # TODO: Add scheduling optimizations (defer copy waits, prefetch pops)
    # for compute-copy overlap. Both this pass and PR #2879's cpu_offload_pass
    # place ao.wait_tensor adjacent to the copy/offload — the compute stream
    # blocks immediately. Deferring the wait to a later point in the forward
    # graph (post-partition) would allow the Triton copy kernel to overlap
    # with subsequent compute. See paged_stashing_guide.md for details.

    gm.graph.lint()
    gm.recompile()

    logger.info(
        "Inserted paged stash ops: %d copy + wait in fwd, %d pop + wait in bwd",
        len(eligible),
        len(eligible),
    )

    # Per-layer breakdown
    layer_counts: dict[str, int] = {}
    for fwd_node, _, _ in eligible:
        fqn = fwd_node.meta.get("custom", {}).get("module_fqn", "")
        parts = fqn.split(".")
        layer_label = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else fqn or "unknown"
        layer_counts[layer_label] = layer_counts.get(layer_label, 0) + 1
    for layer_label in sorted(layer_counts):
        logger.info(
            "  %s: %d stashed activations", layer_label, layer_counts[layer_label]
        )

    return gm


# ---------------------------------------------------------------------------
# aot_fx_trace-compatible pass wrapper
# ---------------------------------------------------------------------------


def paged_stash_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> torch.fx.GraphModule:
    """Paged stash pass for the ``aot_fx_trace`` compile-time pass list.

    Inserts ``paged_stash.copy/pop`` + ``ao.wait_tensor`` ops for eligible
    activations.  Must run **before** ``selective_activation_remat_pass`` so
    that backward consumers are already redirected when SAC + remat runs —
    the original activations have no backward users and are naturally ignored
    by rematerialization.
    """
    return apply_paged_stash_pass(gm, paged_buffers)


# ---------------------------------------------------------------------------
# Memory policies (registered into AVAILABLE_MEMORY_POLICIES)
# ---------------------------------------------------------------------------


def _make_paged_stash_policy(
    must_stash_action: CheckpointPolicy,
) -> callable:
    """Build a SAC policy function that handles MoE fc1 ``_grouped_mm`` nodes.

    Args:
        must_stash_action: What to do with MoE fc1 ``_grouped_mm`` nodes.
            ``PREFER_RECOMPUTE`` when the paged stash pass will redirect
            backward consumers (the activation has no backward users after
            surgery, so remat DCEs it).
            ``MUST_SAVE`` for the baseline experiment that saves expert
            activations as regular tensors (no paged stash).
    """
    save_ops = _get_save_ops()
    save_ops.add(torch.ops._c10d_functional.all_gather_into_tensor.default)

    def policy_fn(node: fx.Node) -> CheckpointPolicy:
        if _is_moe_fc1_grouped_mm(node):
            return must_stash_action
        if node.target in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def paged_stash_tag_policy(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    """Tag nodes for paged activation stashing.

    MoE fc1 ``_grouped_mm`` nodes (identified by op-target + consumer pattern)
    get ``PREFER_RECOMPUTE`` — the paged stash execution pass will redirect
    their backward consumers through paged buffers, so remat naturally DCEs
    the original activations.
    """
    apply_sac_pass(
        gm, policy_fn=_make_paged_stash_policy(CheckpointPolicy.PREFER_RECOMPUTE)
    )
    return gm


def paged_stash_save_only_tag_policy(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    """Baseline: save MoE fc1 ``_grouped_mm`` nodes as regular tensors.

    No paged stash buffers — expert activations are saved in GPU memory
    like any other SAC-saved activation. Useful for measuring the
    fragmentation cost that paged stash exists to solve.
    """
    apply_sac_pass(gm, policy_fn=_make_paged_stash_policy(CheckpointPolicy.MUST_SAVE))
    return gm
