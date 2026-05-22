# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph pass infrastructure for the autoresearch experiment.

This module provides the pass orchestration framework: building the pass list,
applying passes in order, and debug instrumentation. The agent adds custom
pass functions and registers them in ``construct_default_graph_passes``.
"""

from __future__ import annotations

import functools
import operator
import time
from collections.abc import Callable

import torch
from torch._logging import trace_structured

from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.tools.logging import logger


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    use_cudagraph: bool = False,
) -> list[Callable]:
    """Return compile-time passes (used by precompile path)."""
    return construct_default_graph_passes(traced_result, config)


def remove_identity_views(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Peephole pass that removes identity ``aten.view.default`` nodes.

    DTensor plumbing (e.g. ``_api.py:112`` / ``_api.py:229``) emits many
    ``view.default(x, list(x.shape))`` calls — structural no-ops that still
    cost FX/compile time and can confuse downstream fusion. Replace each
    such node with its input.

    Only eliminate when input and target shape can be proven equal without
    symbolic guards; SymInt shapes that can't be statically resolved are
    skipped.
    """
    view_target = torch.ops.aten.view.default
    num_removed = 0
    num_view_nodes = 0
    skipped_no_meta = 0
    skipped_symint = 0
    skipped_shape_mismatch = 0

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not view_target:
            continue
        num_view_nodes += 1

        if len(node.args) < 2:
            continue
        input_node = node.args[0]
        target_size = node.args[1]
        if not isinstance(input_node, torch.fx.Node):
            continue
        val = input_node.meta.get("val", None)
        if val is None or not hasattr(val, "shape"):
            skipped_no_meta += 1
            continue
        input_shape = val.shape
        if len(input_shape) != len(target_size):
            skipped_shape_mismatch += 1
            continue

        equal = True
        for s_in, s_target in zip(input_shape, target_size):
            # Both ints: easy.
            if isinstance(s_in, int) and isinstance(s_target, int):
                if s_in != s_target:
                    equal = False
                    break
                continue
            # Otherwise try identity comparison (same SymInt object) — safe
            # because identical SymNodes mean identical symbolic values
            # without introducing guards.
            if s_in is s_target:
                continue
            # Mixed / non-identical SymInt: skip this view to avoid guards.
            equal = False
            skipped_symint += 1
            break

        if not equal:
            if not isinstance(input_shape[0], int) or any(
                not isinstance(s, int) for s in input_shape
            ):
                # Already accounted for in skipped_symint when applicable.
                pass
            else:
                # Shape mismatch on concrete ints — real reshape, leave alone.
                pass
            continue

        node.replace_all_uses_with(input_node)
        gm.graph.erase_node(node)
        num_removed += 1

    if num_removed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    logger.info(
        f"remove_identity_views: removed {num_removed}/{num_view_nodes} "
        f"aten.view.default nodes "
        f"(skipped: no_meta={skipped_no_meta}, symint={skipped_symint}, "
        f"shape_mismatch={skipped_shape_mismatch})"
    )
    return gm


_INT64_MAX = 9223372036854775807


def remove_identity_ops(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Combined peephole pass for residual no-op patterns left by AOT/DTensor.

    Extends :func:`remove_identity_views` to three more zero-cost patterns:

    1. **Identity slice**: ``aten.slice.Tensor(x, dim, 0, end, 1)`` where
       ``end >= x.shape[dim]`` (incl. ``INT64_MAX``) — replace with ``x``.
    2. **Double transpose**: ``aten.t.default(aten.t.default(x))`` cancels
       to ``x`` (2D tensors). Replace the outer ``t`` with the grandparent.
    3. **Identity dtype cast**: ``aten._to_copy.default(x, dtype=x.dtype,
       ...)`` with no other tensor-attribute change — replace with ``x``.

    These nodes are no-ops in eager but still anchor FX nodes that delay
    deallocation post-AOT. Order matters: slice → double-t → identity cast
    so each step exposes more candidates for the next.
    """
    slice_target = torch.ops.aten.slice.Tensor
    t_target = torch.ops.aten.t.default
    to_copy_target = torch.ops.aten._to_copy.default

    num_slice_total = 0
    num_slice_removed = 0
    num_slice_skipped_meta = 0
    num_slice_skipped_symint = 0

    num_t_total = 0
    num_t_removed = 0

    num_to_copy_total = 0
    num_to_copy_removed = 0
    num_to_copy_skipped_meta = 0

    # ----- Pass 1: identity slice ----------------------------------------
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not slice_target:
            continue
        num_slice_total += 1

        if len(node.args) < 4:
            continue
        input_node = node.args[0]
        dim_arg = node.args[1]
        start = node.args[2]
        end = node.args[3]
        step = node.args[4] if len(node.args) >= 5 else 1
        if step is None:
            step = 1

        if not isinstance(input_node, torch.fx.Node):
            continue
        if not (
            isinstance(dim_arg, int)
            and isinstance(start, int)
            and isinstance(end, int)
            and isinstance(step, int)
        ):
            continue
        if start != 0 or step != 1:
            continue

        val = input_node.meta.get("val", None)
        if val is None or not hasattr(val, "shape"):
            num_slice_skipped_meta += 1
            continue
        shape = val.shape
        if dim_arg < 0:
            dim_arg = dim_arg + len(shape)
        if not (0 <= dim_arg < len(shape)):
            continue
        dim_size = shape[dim_arg]
        if not isinstance(dim_size, int):
            num_slice_skipped_symint += 1
            continue
        # end == INT64_MAX (or any value >= dim_size) means full range.
        if not (end >= dim_size or end == _INT64_MAX):
            continue

        node.replace_all_uses_with(input_node)
        gm.graph.erase_node(node)
        num_slice_removed += 1

    # ----- Pass 2: double transpose --------------------------------------
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not t_target:
            continue
        num_t_total += 1

        if len(node.args) < 1:
            continue
        inner = node.args[0]
        if not isinstance(inner, torch.fx.Node):
            continue
        if inner.op != "call_function" or inner.target is not t_target:
            continue
        if len(inner.args) < 1:
            continue
        grandparent = inner.args[0]
        if not isinstance(grandparent, torch.fx.Node):
            continue

        node.replace_all_uses_with(grandparent)
        gm.graph.erase_node(node)
        num_t_removed += 1
        # Leave `inner` alone — DCE will remove it if it now has no users,
        # otherwise it's still needed by some other path.

    # ----- Pass 3: identity dtype cast -----------------------------------
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not to_copy_target:
            continue
        num_to_copy_total += 1

        if len(node.args) < 1:
            continue
        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue
        val = input_node.meta.get("val", None)
        if val is None or not hasattr(val, "dtype"):
            num_to_copy_skipped_meta += 1
            continue

        target_dtype = node.kwargs.get("dtype", None)
        if target_dtype is not None and target_dtype != val.dtype:
            continue

        # Bail if the cast also changes layout / device / pin_memory in any
        # meaningful way. Treat `None` and a value matching the input as
        # "no change". We only inspect the bound kwargs; missing kwargs are
        # implicitly identity.
        non_identity = False
        target_layout = node.kwargs.get("layout", None)
        if target_layout is not None:
            input_layout = getattr(val, "layout", None)
            if input_layout is not None and target_layout != input_layout:
                non_identity = True

        target_device = node.kwargs.get("device", None)
        if target_device is not None:
            input_device = getattr(val, "device", None)
            if input_device is not None and target_device != input_device:
                non_identity = True

        # pin_memory=True would actually allocate. Only safe if False/None.
        target_pin = node.kwargs.get("pin_memory", None)
        if target_pin:
            non_identity = True

        # memory_format that forces a contiguous re-layout could matter.
        # Only treat as identity if absent or torch.preserve_format.
        target_mf = node.kwargs.get("memory_format", None)
        if target_mf is not None and target_mf is not torch.preserve_format:
            non_identity = True

        if non_identity:
            continue

        node.replace_all_uses_with(input_node)
        gm.graph.erase_node(node)
        num_to_copy_removed += 1

    total_removed = num_slice_removed + num_t_removed + num_to_copy_removed
    if total_removed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    logger.info(
        f"remove_identity_ops: removed "
        f"slice={num_slice_removed}/{num_slice_total} "
        f"(skipped: no_meta={num_slice_skipped_meta}, "
        f"symint={num_slice_skipped_symint}), "
        f"double_t={num_t_removed}/{num_t_total}, "
        f"identity_cast={num_to_copy_removed}/{num_to_copy_total} "
        f"(skipped: no_meta={num_to_copy_skipped_meta})"
    )
    return gm


def elide_split_cat_for_reduce_scatter(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Replace ``cat([split(x, ...)], dim=0)`` with a no-copy ``view`` when the
    cat output is consumed by a ``reduce_scatter_tensor``.

    Every TP-sharded matmul (wo / w2 / lm_head / tok_embeddings) emits the
    pattern

    .. code-block:: python

        s_i = aten.split.Tensor(x, split_size, split_dim)          # views
        y   = aten.cat.default([s_0, ..., s_{N-1}])                # dim=0, alloc
        z   = reduce_scatter_tensor(y, 'sum', world_size, group)

    where ``x`` is contiguous with shape ``[..., N*split_size, ...]`` along
    ``split_dim`` and the cat stacks the resulting views back along a new
    outer axis (``dim=0``). When ``split_dim`` is the *second* tensor dim
    (so collapsing it produces an extra leading dim) and ``x`` is
    contiguous, the cat is byte-equivalent to viewing ``x`` as the cat's
    output shape — no data movement is required.

    Concretely, for the high-volume Llama3 8B TP=2 case
    ``x : bf16[1, 8192, 4096][33554432, 4096, 1]`` with
    ``split(x, 4096, 1)`` followed by ``cat([s0, s1])`` (dim=0 default)
    produces ``[2, 4096, 4096][16777216, 4096, 1]`` whose byte layout is
    identical to ``x`` viewed as ``[2, 4096, 4096]``: dropping the cat
    saves a 64 MiB allocation and copy per occurrence (~130/step on
    Llama3 8B).
    """
    cat_target = torch.ops.aten.cat.default
    split_target = torch.ops.aten.split.Tensor
    rs_target = torch.ops._c10d_functional.reduce_scatter_tensor.default
    view_target = torch.ops.aten.view.default

    num_cat_total = 0
    num_elided = 0
    skipped_not_dim0 = 0
    skipped_no_rs_consumer = 0
    skipped_not_pure_split = 0
    skipped_non_contiguous = 0
    skipped_no_meta = 0
    skipped_symint = 0
    skipped_shape_mismatch = 0

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not cat_target:
            continue
        num_cat_total += 1

        # aten.cat.default(tensors, dim=0) — dim is args[1] when present,
        # otherwise the default 0.
        cat_dim = node.args[1] if len(node.args) >= 2 else 0
        if cat_dim != 0:
            skipped_not_dim0 += 1
            continue

        cat_inputs = node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) == 0:
            continue

        # All cat inputs must be getitem nodes pointing at the same split.
        split_node = None
        getitem_indices: list[int] = []
        ok = True
        for inp in cat_inputs:
            if not isinstance(inp, torch.fx.Node):
                ok = False
                break
            if inp.op != "call_function" or inp.target is not operator.getitem:
                ok = False
                break
            parent = inp.args[0]
            if not isinstance(parent, torch.fx.Node):
                ok = False
                break
            if parent.op != "call_function" or parent.target is not split_target:
                ok = False
                break
            if split_node is None:
                split_node = parent
            elif parent is not split_node:
                ok = False
                break
            idx = inp.args[1]
            if not isinstance(idx, int):
                ok = False
                break
            getitem_indices.append(idx)
        if not ok or split_node is None:
            skipped_not_pure_split += 1
            continue

        # Split args: (x, split_size, split_dim=0)
        if len(split_node.args) < 2:
            skipped_not_pure_split += 1
            continue
        x_node = split_node.args[0]
        if not isinstance(x_node, torch.fx.Node):
            skipped_not_pure_split += 1
            continue
        split_dim = split_node.args[2] if len(split_node.args) >= 3 else 0
        if not isinstance(split_dim, int):
            skipped_not_pure_split += 1
            continue

        x_val = x_node.meta.get("val", None)
        cat_val = node.meta.get("val", None)
        if (
            x_val is None
            or cat_val is None
            or not hasattr(x_val, "shape")
            or not hasattr(cat_val, "shape")
        ):
            skipped_no_meta += 1
            continue

        x_shape = list(x_val.shape)
        cat_shape = list(cat_val.shape)

        # Only handle concrete int shapes; SymInt is out of scope.
        if any(not isinstance(s, int) for s in x_shape) or any(
            not isinstance(s, int) for s in cat_shape
        ):
            skipped_symint += 1
            continue

        # Geometry equivalence: cat([split(x, k, d)], dim=0) is byte-identical
        # to view(x, cat_shape) iff
        #   1. x is contiguous
        #   2. split_dim is the *second* tensor dim (index 1) and ranks before
        #      it are all size 1 — i.e. x[:k, :] type shapes — so collapsing
        #      that axis into an outer N=axis-size/k just reshapes contiguous
        #      bytes.  More generally, the prefix dims of x before split_dim
        #      must all be size 1 so that the cat's new leading dim
        #      effectively replaces them.
        #   3. The number of splits equals cat's leading dim.
        #   4. The trailing shape after split_dim matches cat_shape[1+...].
        if split_dim < 1 or split_dim >= len(x_shape):
            skipped_shape_mismatch += 1
            continue
        if any(s != 1 for s in x_shape[:split_dim]):
            skipped_shape_mismatch += 1
            continue

        num_splits = len(cat_inputs)
        # cat output shape should be [num_splits, split_size, *x_shape[split_dim+1:]]
        # where split_size = x_shape[split_dim] // num_splits exactly.
        axis_size = x_shape[split_dim]
        if num_splits == 0 or axis_size % num_splits != 0:
            skipped_shape_mismatch += 1
            continue
        split_size = axis_size // num_splits
        # split_node.args[1] is the declared split size; it must match.
        if not isinstance(split_node.args[1], int) or split_node.args[1] != split_size:
            skipped_shape_mismatch += 1
            continue
        # Expected cat shape: [num_splits, split_size, *x_shape[split_dim+1:]]
        expected_cat_shape = [num_splits, split_size] + x_shape[split_dim + 1 :]
        if cat_shape != expected_cat_shape:
            skipped_shape_mismatch += 1
            continue
        # The getitems should consume the split in increasing index order
        # (cat stacks them in input order).
        if getitem_indices != list(range(num_splits)):
            skipped_shape_mismatch += 1
            continue

        # Contiguity check on the split source: only valid when x's byte
        # layout matches what `view(cat_shape)` would expect.
        try:
            x_is_contig = bool(x_val.is_contiguous())
        except Exception:
            x_is_contig = False
        if not x_is_contig:
            skipped_non_contiguous += 1
            continue

        # The cat must feed a reduce_scatter (single consumer).
        cat_users = list(node.users)
        if len(cat_users) != 1:
            skipped_no_rs_consumer += 1
            continue
        rs = cat_users[0]
        if rs.op != "call_function" or rs.target is not rs_target:
            skipped_no_rs_consumer += 1
            continue

        # Perform the rewrite: insert view(x, cat_shape) just before the cat.
        with gm.graph.inserting_before(node):
            new_view = gm.graph.call_function(
                view_target, args=(x_node, list(cat_shape))
            )
        new_view.meta["val"] = cat_val
        # Preserve source-location / annotations from the original cat so
        # downstream tooling (tlparse, annotations) keeps attribution.
        if "stack_trace" in node.meta:
            new_view.meta["stack_trace"] = node.meta["stack_trace"]
        if "custom" in node.meta:
            new_view.meta["custom"] = node.meta["custom"]

        node.replace_all_uses_with(new_view)
        gm.graph.erase_node(node)
        num_elided += 1
        # DCE will reclaim the split + getitems if they have no other users;
        # if any getitem has additional consumers, the split stays alive.

    if num_elided:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    # Saved memory traffic: each elided cat avoids ``2 * cat_shape.numel() *
    # dtype.itemsize`` bytes of alloc+copy (read+write). At 130 cats × 64 MiB
    # each in the Llama3 8B TP=2 case, that's ~8.3 GiB per step.
    logger.info(
        f"elide_split_cat_for_reduce_scatter: elided {num_elided}/{num_cat_total} "
        f"split-cat-reduce_scatter cats (skipped: not_dim0={skipped_not_dim0}, "
        f"not_pure_split={skipped_not_pure_split}, "
        f"no_rs_consumer={skipped_no_rs_consumer}, "
        f"non_contiguous={skipped_non_contiguous}, "
        f"shape_mismatch={skipped_shape_mismatch}, "
        f"no_meta={skipped_no_meta}, symint={skipped_symint})"
    )
    return gm


def bucket_fsdp_collectives(
    gm: torch.fx.GraphModule,
    example_inputs=None,
    *,
    max_bucket_size: int = 9,
) -> torch.fx.GraphModule:
    """Coalesce contiguous FSDP all-gathers (forward) and reduce-scatters
    (backward) into ``*_coalesced`` collective calls.

    FSDP at world_size=4 emits ~291 ``all_gather_into_tensor`` calls
    (forward weight gathers) and ~291 ``reduce_scatter_tensor`` calls
    (backward param-grad reductions) per step. NCCL launch overhead per
    tiny tensor (e.g. RMSNorm γ at a few KiB) is non-trivial. The
    functional collective registry exposes coalesced variants

    .. code-block:: python

        _c10d_functional.all_gather_into_tensor_coalesced(
            Tensor[] inputs, int group_size, Any group_name) -> Tensor[]
        _c10d_functional.reduce_scatter_tensor_coalesced(
            Tensor[] inputs, str reduce_op, int group_size, Any group_name) -> Tensor[]

    which issue a single collective for an entire batch of same-direction,
    same-group inputs. We coalesce contiguous runs (in graph node order)
    that share ``(world_size, group_name)`` and don't have any earlier
    bucket member's ``wait_tensor`` consumed inside the bucket window.
    The latter constraint avoids delaying a wait that is on the critical
    path between two candidate collectives.

    Each bucket replaces N collective calls with one coalesced call plus
    N ``operator.getitem`` extractions. Existing ``wait_tensor`` nodes are
    rewired to consume the matching getitem. Caps at ``max_bucket_size`` to
    keep the working set per collective bounded.

    Numerics are bitwise-identical to issuing the collectives separately —
    a coalesced call performs the same per-tensor reduction/gather.
    """
    ag_target = torch.ops._c10d_functional.all_gather_into_tensor.default
    rs_target = torch.ops._c10d_functional.reduce_scatter_tensor.default
    wait_target = torch.ops._c10d_functional.wait_tensor.default

    have_ag_coalesced = hasattr(
        torch.ops._c10d_functional, "all_gather_into_tensor_coalesced"
    )
    have_rs_coalesced = hasattr(
        torch.ops._c10d_functional, "reduce_scatter_tensor_coalesced"
    )
    ag_coalesced_target = (
        torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
        if have_ag_coalesced
        else None
    )
    rs_coalesced_target = (
        torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default
        if have_rs_coalesced
        else None
    )

    # Cache node positions for graph-position queries.
    node_pos: dict[torch.fx.Node, int] = {
        n: i for i, n in enumerate(gm.graph.nodes)
    }

    def _wait_node(c: torch.fx.Node) -> torch.fx.Node | None:
        """Find the wait_tensor consumer of a collective node (single user expected)."""
        users = list(c.users)
        if len(users) != 1:
            return None
        w = users[0]
        if w.op != "call_function" or w.target is not wait_target:
            return None
        return w

    def _eligible(c: torch.fx.Node, world_size_arg_idx: int, ws_target: int) -> bool:
        if len(c.args) <= world_size_arg_idx:
            return False
        ws = c.args[world_size_arg_idx]
        if not isinstance(ws, int) or ws != ws_target:
            return False
        # Must have meta["val"] with concrete shape.
        val = c.meta.get("val", None)
        if val is None or not hasattr(val, "shape"):
            return False
        for s in val.shape:
            if not isinstance(s, int):
                return False
        # Input must also have concrete meta.
        if not isinstance(c.args[0], torch.fx.Node):
            return False
        in_val = c.args[0].meta.get("val", None)
        if in_val is None or not hasattr(in_val, "shape"):
            return False
        for s in in_val.shape:
            if not isinstance(s, int):
                return False
        # Must have a single wait_tensor user (no other consumers of the
        # raw collective output).
        if _wait_node(c) is None:
            return False
        return True

    def _wait_is_simple(w: torch.fx.Node) -> bool:
        """The wait must have at least one downstream user; bucketing
        doesn't change the wait result's value or layout, only its
        producer."""
        return len(list(w.users)) >= 1

    def _private_producer_chain(
        c: torch.fx.Node,
    ) -> list[torch.fx.Node] | None:
        """Walk upstream from a collective's input, collecting nodes whose
        users are all within the visited frontier or the collective itself.

        Returns the producer subgraph in topological order, or ``None`` if
        the chain is not "private" (i.e. some node has external users that
        would be invalidated by moving the chain).

        For FSDP AGs in this graph the chain is typically a single
        ``_to_copy(placeholder)`` cast, which is trivially private.
        """
        in_node = c.args[0]
        if not isinstance(in_node, torch.fx.Node):
            return None
        chain: list[torch.fx.Node] = []
        visited: set[torch.fx.Node] = set()
        # BFS upstream, only following nodes whose users are subset of
        # {c, ...chain so far}.
        frontier = [in_node]
        while frontier:
            n = frontier.pop()
            if n in visited:
                continue
            if n.op == "placeholder" or n.op == "get_attr":
                # Stop walking upstream; the placeholder is global and
                # doesn't need to be moved.
                continue
            users_set = set(n.users)
            allowed = {c} | visited | {in_node}
            # `n` itself may not yet be in visited. Allow chain so far.
            if not users_set.issubset(allowed):
                return None
            visited.add(n)
            chain.append(n)
            for inp in n.all_input_nodes:
                if inp not in visited:
                    frontier.append(inp)
        # Sort chain by current graph position so we can move in order.
        chain.sort(key=lambda x: node_pos.get(x, 0))
        return chain

    def _build_buckets_with_hoist(
        nodes: list[torch.fx.Node],
        world_size_arg_idx: int,
        group_name_arg_idx: int,
        hoist_producers: bool,
    ) -> tuple[list[list[torch.fx.Node]], dict[torch.fx.Node, list[torch.fx.Node]]]:
        """Group eligible collective nodes into buckets.

        If ``hoist_producers`` is True, the producer chain of each member
        is required to be private (cast-from-placeholder pattern). We can
        then move all chains to a common insertion point and don't need
        the critical-path constraint between members.

        If ``hoist_producers`` is False, we apply the critical-path
        constraint: no consumer of an earlier-bucket-member wait may lie
        between bucket members. This protects waits that are immediately
        consumed by compute.
        """
        buckets: list[list[torch.fx.Node]] = []
        chains: dict[torch.fx.Node, list[torch.fx.Node]] = {}
        current: list[torch.fx.Node] = []
        current_waits: list[torch.fx.Node] = []
        current_group: object | None = None

        for c in nodes:
            group = c.args[group_name_arg_idx]
            w = _wait_node(c)
            assert w is not None
            if not _wait_is_simple(w):
                if current:
                    buckets.append(current)
                current, current_waits, current_group = [], [], None
                continue
            chain = None
            if hoist_producers:
                chain = _private_producer_chain(c)
                if chain is None:
                    # Not hoistable; close current bucket and skip.
                    if current:
                        buckets.append(current)
                    current, current_waits, current_group = [], [], None
                    continue
            start_new = False
            if not current:
                start_new = True
            elif len(current) >= max_bucket_size:
                start_new = True
            elif group != current_group:
                start_new = True
            elif not hoist_producers:
                last_pos = node_pos[current[-1]]
                c_pos = node_pos[c]
                conflict = False
                for prev_w in current_waits:
                    for u in prev_w.users:
                        u_pos = node_pos.get(u, None)
                        if u_pos is None:
                            continue
                        if last_pos < u_pos < c_pos:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    start_new = True

            if start_new:
                if current:
                    buckets.append(current)
                current, current_waits = [c], [w]
                current_group = group
            else:
                current.append(c)
                current_waits.append(w)
            if chain is not None:
                chains[c] = chain

        if current:
            buckets.append(current)
        return [b for b in buckets if len(b) >= 2], chains

    # --- Forward: all_gather_into_tensor (ws=4) -------------------------
    ag_candidates: list[torch.fx.Node] = []
    skipped_ag_ineligible = 0
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target is not ag_target:
            continue
        # AG signature: (Tensor input, int group_size, Any group_name).
        if _eligible(n, world_size_arg_idx=1, ws_target=4):
            ag_candidates.append(n)
        else:
            skipped_ag_ineligible += 1

    ag_buckets: list[list[torch.fx.Node]] = []
    ag_chains: dict[torch.fx.Node, list[torch.fx.Node]] = {}
    if have_ag_coalesced and ag_candidates:
        ag_buckets, ag_chains = _build_buckets_with_hoist(
            ag_candidates,
            world_size_arg_idx=1,
            group_name_arg_idx=2,
            hoist_producers=True,
        )

    # --- Backward: reduce_scatter_tensor (ws=4) -------------------------
    rs_candidates: list[torch.fx.Node] = []
    skipped_rs_ineligible = 0
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target is not rs_target:
            continue
        # RS signature: (Tensor input, str reduce_op, int group_size, Any group_name).
        if _eligible(n, world_size_arg_idx=2, ws_target=4):
            rs_candidates.append(n)
        else:
            skipped_rs_ineligible += 1

    rs_buckets: list[list[torch.fx.Node]] = []
    if have_rs_coalesced and rs_candidates:
        rs_buckets, _ = _build_buckets_with_hoist(
            rs_candidates,
            world_size_arg_idx=2,
            group_name_arg_idx=3,
            hoist_producers=False,
        )

    def _rewrite_bucket(
        bucket: list[torch.fx.Node],
        coalesced_target,
        extra_args: tuple,
        chains: dict[torch.fx.Node, list[torch.fx.Node]] | None = None,
    ) -> int:
        """Insert a coalesced collective for the bucket and rewire waits.

        Two modes:

        - **No hoist** (``chains is None``): anchor the coalesced call
          right before the LAST bucket member. All inputs precede the
          last member, so they're defined. Waits are moved to right after
          their getitems; per the critical-path constraint built at
          bucket-construction time, no consumer of any bucket wait lies
          before the last member, so this is safe.

        - **With hoist** (``chains`` provided): anchor the coalesced call
          right before the FIRST bucket member. Producer chains for
          members 2..N (typically a single ``_to_copy`` cast from a
          placeholder) are moved to just before that anchor so all
          inputs are defined. Waits move forward (earlier) — safe since
          the original wait positions were AFTER the first member.
        """
        inputs = [c.args[0] for c in bucket]
        waits = [_wait_node(c) for c in bucket]
        out_vals = [c.meta["val"] for c in bucket]
        if chains is None:
            anchor = bucket[-1]
        else:
            anchor = bucket[0]
            # Hoist the producer chain of each bucket member to just
            # before the anchor, in topological order. The first
            # member's chain is already before the anchor by definition,
            # but moving idempotent. Use `anchor.prepend(node)` to place
            # each chain node immediately before the anchor; chain
            # members are processed in dependency order so transitive
            # dependencies precede their dependents.
            for c in bucket[1:]:
                chain = chains.get(c, [])
                for node in chain:
                    # `prepend` inserts before; chain is in topo order so
                    # processing in order places earlier nodes earlier.
                    anchor.prepend(node)

        with gm.graph.inserting_before(anchor):
            coalesced = gm.graph.call_function(
                coalesced_target,
                args=(inputs,) + extra_args,
            )
        coalesced.meta["val"] = list(out_vals)
        prev = coalesced
        getitems: list[torch.fx.Node] = []
        for i, ov in enumerate(out_vals):
            with gm.graph.inserting_after(prev):
                gi = gm.graph.call_function(
                    operator.getitem, args=(coalesced, i)
                )
            gi.meta["val"] = ov
            getitems.append(gi)
            prev = gi
        for c, w, gi in zip(bucket, waits, getitems):
            assert w is not None
            gi.append(w)
            w.args = (gi,) + w.args[1:]
        return len(bucket)

    # --- Rewrite forward AG buckets ------------------------------------
    num_ag_coalesced = 0
    for bucket in ag_buckets:
        first = bucket[0]
        group_size = first.args[1]
        group_name = first.args[2]
        num_ag_coalesced += _rewrite_bucket(
            bucket, ag_coalesced_target, (group_size, group_name), ag_chains
        )

    # --- Rewrite backward RS buckets -----------------------------------
    num_rs_coalesced = 0
    for bucket in rs_buckets:
        first = bucket[0]
        reduce_op = first.args[1]
        group_size = first.args[2]
        group_name = first.args[3]
        if any(c.args[1] != reduce_op for c in bucket):
            continue
        num_rs_coalesced += _rewrite_bucket(
            bucket, rs_coalesced_target, (reduce_op, group_size, group_name)
        )

    if num_ag_coalesced or num_rs_coalesced:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    ag_bucket_sizes = [len(b) for b in ag_buckets]
    rs_bucket_sizes = [len(b) for b in rs_buckets]
    ag_avg = (sum(ag_bucket_sizes) / len(ag_bucket_sizes)) if ag_bucket_sizes else 0.0
    rs_avg = (sum(rs_bucket_sizes) / len(rs_bucket_sizes)) if rs_bucket_sizes else 0.0
    logger.info(
        f"bucket_fsdp_collectives: AG ws=4 candidates={len(ag_candidates)}, "
        f"buckets={len(ag_buckets)}, coalesced={num_ag_coalesced}, "
        f"avg_size={ag_avg:.2f}, op_available={have_ag_coalesced}, "
        f"skipped_ineligible={skipped_ag_ineligible} | "
        f"RS ws=4 candidates={len(rs_candidates)}, "
        f"buckets={len(rs_buckets)}, coalesced={num_rs_coalesced}, "
        f"avg_size={rs_avg:.2f}, op_available={have_rs_coalesced}, "
        f"skipped_ineligible={skipped_rs_ineligible}"
    )
    return gm


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    The agent adds custom passes to this list.
    """
    passes: list[Callable] = [
        remove_identity_views,
        remove_identity_ops,
        elide_split_cat_for_reduce_scatter,
        bucket_fsdp_collectives,
    ]
    return passes


def _get_pass_name(pass_fn: Callable) -> str:
    return (
        pass_fn.func.__name__
        if isinstance(pass_fn, functools.partial)
        else pass_fn.__name__
    )


def _filter_disabled_passes(
    passes: list[Callable], disable_names: list[str]
) -> list[Callable]:
    """Remove passes whose names exactly match any entry in ``disable_names``."""
    disable_set = set(disable_names)
    filtered = []
    skipped = []
    for pass_fn in passes:
        name = _get_pass_name(pass_fn)
        if name in disable_set:
            skipped.append(name)
        else:
            filtered.append(pass_fn)
    if skipped:
        logger.info(f"Disabled {len(skipped)} graph passes: {skipped}")
    return filtered


def apply_graph_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    passes: list[Callable],
    *,
    compile_config: "GraphTrainerCompileConfig | None" = None,
) -> torch.fx.GraphModule:
    """Apply graph passes to the traced fwd+bwd graph.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example (fake) inputs matching the graph signature.
        passes: Ordered list of pass callables, each with signature
            ``(gm, example_inputs, **kwargs) -> gm``.
        compile_config: Optional compile config. When provided and
            ``debug_graph_passes`` is True, logs timing, op-count diffs,
            and before/after graphs to tlparse for each pass.
    """
    debug = compile_config is not None and compile_config.debug_graph_passes
    disable_patterns = (
        compile_config.disable_passes if compile_config is not None else []
    )
    if disable_patterns:
        passes = _filter_disabled_passes(passes, disable_patterns)
    pass_names = [_get_pass_name(pass_fn) for pass_fn in passes]
    pass_list = "\n  ".join(f"{i}. {name}" for i, name in enumerate(pass_names, 1))
    logger.info(f"Applying {len(passes)} graph passes:\n  {pass_list}")
    all_passes_start = time.perf_counter()
    tlparse_log_graph_pass(gm, graph_name="make_fx_graph_traced", debug=debug)
    for pass_fn in passes:
        pass_name = _get_pass_name(pass_fn)
        if debug:
            tlparse_log_graph_pass(gm, graph_name=f"before_{pass_name}", debug=debug)
            before_snapshot = snapshot_graph(gm)
            start = time.perf_counter()
        gm = pass_fn(gm, example_inputs)
        assert isinstance(
            gm, torch.fx.GraphModule
        ), f"Pass {pass_name} returned {type(gm).__name__}, expected GraphModule"
        if debug:
            elapsed = time.perf_counter() - start
            logger.info(f"Pass {pass_name} took {elapsed:.3f}s")
            tlparse_log_graph_pass(gm, graph_name=f"after_{pass_name}", debug=debug)
            after_snapshot = snapshot_graph(gm)
            log_graph_diff(before_snapshot, after_snapshot, pass_name)
    all_passes_elapsed = time.perf_counter() - all_passes_start
    logger.info(f"All {len(passes)} graph passes took {all_passes_elapsed:.3f}s")
    return gm


def tlparse_log_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    graph_name: str,
    debug: bool = False,
) -> torch.fx.GraphModule:
    """Log the transformed graph to tlparse via trace_structured."""
    additional_meta = ["autograd_backward"]
    if debug:
        additional_meta.append("seq_nr")

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
            additional_meta=additional_meta,
        ),
        expect_trace_id=False,
    )

    return gm
