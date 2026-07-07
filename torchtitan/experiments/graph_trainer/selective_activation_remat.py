# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AC rematerialize pass: in-place duplicate recompute nodes for backward."""

import logging
from typing import Any

import torch
import torch.fx as fx
from torch._functorch.compile_utils import raise_getitems
from torch._functorch.partitioners import (
    has_recomputable_ops,
    has_recomputable_rng_ops,
    must_recompute,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    _is_backward_node,
)


log = logging.getLogger(__name__)


def _privatize_custom_meta(node: fx.Node) -> None:
    """Give ``node`` its own ``meta["custom"]`` dict.

    ``fx.Graph.node_copy`` / ``dict.update`` shallow-copy ``meta``, so a copied node
    ends up sharing its source's nested ``custom`` dict; annotating one would then
    silently mutate the other. A shallow copy suffices -- callers only add or
    replace keys, they don't mutate the values in place.
    """
    custom = node.meta.get("custom")
    if custom is not None:
        node.meta["custom"] = dict(custom)


# Module FQNs whose backward is not an activation-checkpoint region. In a
# forward-loss-backward graph the backward runs in reverse module order, so the
# loss / lm_head backward always precedes the model-layer backward. A
# chunked-loss / lm_head backward can *consume* must_recompute forward nodes
# (the tied embedding weight and shared rope buffers flow into the loss
# backward), but recompute belongs to the model-layer backward. These FQNs mark
# where that model-layer backward begins.
_NON_AC_MODULE_FQNS = frozenset({"loss", "lm_head"})


def _collect_backward_regions(
    gm: fx.GraphModule,
) -> list[tuple[int, int, bool]]:
    """Returns (bwd_start, bwd_end, needs_remat) for each model-layer backward region.

    Relies on the forward-loss-backward structure: the loss / lm_head backward
    always precedes the model-layer backward. We find the last loss / lm_head
    node and collect backward regions only from there on, so chunked-loss
    backward chunks are never counted as remat regions -- only the model-layer
    backward that follows is. When there are no loss / lm_head nodes (e.g. a
    synthetic unit-test graph) collection starts at the beginning.

    Regions are maximal contiguous runs of backward nodes, as [start, end)
    indices into the graph node list. ``needs_remat`` is True when a backward
    node in the region consumes a must_recompute forward node.
    """
    all_nodes = list(gm.graph.nodes)

    # Skip everything up to and including the last loss / lm_head node.
    start_idx = 0
    for idx, node in enumerate(all_nodes):
        if _get_module_fqn(node) in _NON_AC_MODULE_FQNS:
            start_idx = idx + 1

    regions: list[tuple[int, int, bool]] = []
    bwd_start: int | None = None
    needs_remat = False

    for idx in range(start_idx, len(all_nodes)):
        node = all_nodes[idx]
        if _is_backward_node(node):
            if bwd_start is None:
                bwd_start = idx
                needs_remat = False
            if not needs_remat and any(
                must_recompute(inp) for inp in node.all_input_nodes
            ):
                needs_remat = True
        elif bwd_start is not None:
            regions.append((bwd_start, idx, needs_remat))
            bwd_start = None

    if bwd_start is not None:
        regions.append((bwd_start, len(all_nodes), needs_remat))

    return regions


def selective_activation_remat_pass(
    gm: fx.GraphModule,
    example_inputs: Any = None,
) -> fx.GraphModule:
    """In-place remat: insert recompute duplicates before backward consumers.

    For each ``must_recompute`` forward node consumed by a backward node, a
    duplicate is inserted just before the backward consumer and that
    consumer's args are redirected to the duplicate. Original forward nodes
    whose consumers were all backward become dead and are erased; originals
    with remaining forward consumers stay.

    The graph is mutated in place: original node identities and names are
    preserved, only the duplicates carry a ``_recomputed`` suffix. No
    whole-graph DCE or topological reorder is performed.

    Backward regions are identified by
    ``node.meta["autograd_backward"] == True``, set by both Dynamo and
    non-strict ``make_fx`` tracing when tracing ``torch.autograd.grad``.

    The graph may contain multiple disjoint backward regions (e.g. one per
    chunk of a chunked loss). ``_collect_backward_regions`` excludes loss /
    lm_head regions from remat even when they consume must_recompute forward
    nodes, so recompute is driven only by the model-layer backward. Exactly one
    remat region is expected; more than one is an error.
    """
    if not has_recomputable_ops(gm):
        return gm

    if has_recomputable_rng_ops(gm):
        raise RuntimeError(
            "Activation checkpoint rematerialization in `forward-loss-backward` graph does not support RNG ops "
            "in recompute regions. Please move RNG operations outside "
            "of recompute regions, or use joint graph mode (where partitioner handles RNG)."
        )

    regions = _collect_backward_regions(gm)
    if not regions:
        return gm

    # Loss / lm_head backward regions are already filtered out by
    # _collect_backward_regions, so the remaining needs-remat regions are the
    # model-layer AC regions; exactly one is expected.
    remat_regions = [(s, e) for s, e, needs in regions if needs]

    if len(remat_regions) > 1:
        raise RuntimeError(
            f"Detected {len(remat_regions)} disjoint backward regions that require "
            "recomputation, but remat supports only one such region in a "
            "forward-loss-backward graph."
        )

    if not remat_regions:
        return gm

    bwd_start, bwd_end = remat_regions[0]

    all_nodes = list(gm.graph.nodes)
    bwd_nodes = all_nodes[bwd_start:bwd_end]
    order = {n: i for i, n in enumerate(all_nodes)}

    # Map each must_recompute fwd node to the bwd node its dup will be
    # inserted in front of. The earliest bwd consumer (in graph order)
    # wins via ``setdefault`` below.
    remat_targets: dict[fx.Node, fx.Node] = {}

    def collect_fw_nodes_to_recompute_for(bwd_node: fx.Node) -> None:
        seen: set[fx.Node] = set()

        def _gather(n: fx.Node) -> None:
            if n in seen or not must_recompute(n):
                return
            seen.add(n)
            remat_targets.setdefault(n, bwd_node)
            for inp in n.all_input_nodes:
                _gather(inp)

        # bwd_node itself may not be must_recompute; start from its inputs.
        for inp in bwd_node.all_input_nodes:
            _gather(inp)

    for bwd_node in bwd_nodes:
        collect_fw_nodes_to_recompute_for(bwd_node)

    # Map original forward must_recompute node -> its recomputed duplicate.
    recomputed_nodes: dict[fx.Node, fx.Node] = {}
    # CPU offload: track which bwd target each reload-chain node was last
    # hoisted before, so we can re-hoist if an earlier dup needs it later.
    moved_offload: dict[fx.Node, fx.Node] = {}

    # Build offloaded_fwd -> bwd_wait map by walking the offload op pattern
    # (apply_cpu_offload_pass emits: F -> ao.offload -> ao.wait_tensor ->
    # ao.reload -> ao.wait_tensor). Used to redirect a recompute dup that
    # consumes an offloaded fwd to read from the bwd-region GPU value.
    offloaded_fwd_to_bwd_wait: dict[fx.Node, fx.Node] = {}
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.ao.offload.default
    ):
        offloaded_fwd = node.args[0]
        fwd_wait = next(
            (u for u in node.users if u.target is torch.ops.ao.wait_tensor.default),
            None,
        )
        if fwd_wait is None:
            continue
        reload_op = next(
            (u for u in fwd_wait.users if u.target is torch.ops.ao.reload.default),
            None,
        )
        if reload_op is None:
            continue
        bwd_wait = next(
            (
                u
                for u in reload_op.users
                if u.target is torch.ops.ao.wait_tensor.default
            ),
            None,
        )
        if bwd_wait is None:
            continue
        offloaded_fwd_to_bwd_wait[offloaded_fwd] = bwd_wait

    def ensure_offload_chain_before(reload_node: fx.Node, target: fx.Node) -> None:
        """Move ``reload_node`` and its bwd-region deps in front of ``target``.

        A recompute dup consuming an offloaded forward node must read from
        the reload chain on GPU, not from F's freed storage. The offload
        pass places the reload chain before F's *first existing* backward
        consumer, but a recompute dup is a NEW consumer that the offload
        pass didn't see. The dup may land earlier in graph order than F's
        first existing backward consumer; when it does, the chain must be
        hoisted to keep ``dup -> reload_chain`` topologically valid.

        Concrete example where this fires (layer K's offloaded residual,
        consumed in forward by layer K+1 via a must_recompute op):

            # forward (layer K):
            F = add(...)                           # offloaded
            offload_op = ao.offload(F); ...        # → CPU, frees F's GPU mem
            # forward (layer K+1):
            N = layer_norm(F)                      # must_recompute, needs F

            # ──────── backward begins ────────
            # backward of layer K+1 (runs FIRST in reverse-order bwd):
            grad_layer_K1_input = ...              # transitively wants N
            #   ↑ remat inserts N_recomputed here  ← bwd_target for N
            #     N_recomputed's arg F is redirected to wait_tensor

            # backward of layer K (runs LATER in bwd):
            reload_default = ao.reload(...)        # ← chain originally here:
            wait_tensor    = ao.wait_tensor(...)   #   offload pass placed it
            grad_F = layer_K_bwd(wait_tensor, ...) #   before *this* consumer

        Because backward is reverse, layer K+1's backward is *earlier* in
        graph order than layer K's backward — so N_recomputed sits ahead
        of the reload chain. Without hoisting, N_recomputed would
        reference a wait_tensor that hasn't been defined yet → topology
        violation. We move the chain in front of ``target`` (here: the
        first bwd_node of layer K+1's backward).

        ``moved_offload`` keeps moves idempotent and ensures the chain
        ends up before the earliest target across repeated calls.
        """
        # ``bwd_reload_chain`` is the set of backward-region nodes that
        # need to relocate together: ``reload_node`` (typically the
        # ``ao.wait_tensor`` whose value the dup will read) plus every
        # backward-region node it transitively depends on (typically the
        # ``ao.reload`` op feeding it). Forward-region deps stop the walk —
        # they're already before any backward target.
        bwd_reload_chain: set[fx.Node] = set()
        stack = [reload_node]
        while stack:
            n = stack.pop()
            if n in bwd_reload_chain or not _is_backward_node(n):
                continue
            # Skip if n is already in front of ``target``: either previously
            # hoisted before an earlier-or-equal target, or sitting at its
            # original (offload-pass) position which already precedes target.
            # Re-hoisting in either case would collapse the prefetch gap the
            # offload pass set up, killing async H2D overlap.
            prev = moved_offload.get(n)
            anchor_pos = order[prev] if prev is not None else order[n]
            if anchor_pos <= order[target]:
                continue
            bwd_reload_chain.add(n)
            stack.extend(n.all_input_nodes)

        # TODO: when we DO move (chain is currently behind target), all
        # chain members get prepended adjacent to ``target`` — collapsing
        # the prefetch gap between ``reload`` and ``wait_tensor`` to ~0
        # and serializing the H2D against compute. If this becomes a
        # measurable regression we should place ``reload`` early in the
        # bwd region and ``wait_tensor`` just before ``target`` to restore
        # overlap.
        # Prepend in graph (topological) order so deps land before dependents.
        for n in sorted(bwd_reload_chain, key=order.__getitem__):
            target.prepend(n)
            moved_offload[n] = target
            log.debug("moved %s before %s", n.name, target.name)

    def remat_input(x: object) -> object:
        """Arg-transform: redirect must_recompute originals to their dups, and
        offloaded forward nodes to their CPU-reload chain. Hoisting of the
        reload chain happens separately in the dup-creation loop."""
        if not isinstance(x, fx.Node):
            return x
        if x in recomputed_nodes:
            return recomputed_nodes[x]
        bwd_wait = offloaded_fwd_to_bwd_wait.get(x)
        if bwd_wait is not None:
            return bwd_wait
        return x

    def duplicate_subgraph_get_attrs(
        fwd_node: fx.Node, bwd_target: fx.Node
    ) -> dict[fx.Node, fx.Node]:
        """Give a recompute dup private copies of its subgraph get_attr inputs.

        HOPs like ``flex_attention`` reference their ``score_mod`` / ``mask_mod``
        subgraphs through ``get_attr`` nodes. If the dup shared the original's
        get_attr, the later ``regional_inductor`` pass — which partitions the
        annotated flex node together with its subgraph get_attrs — could only
        place that shared get_attr in one flex region's partition, leaving the
        other flex with the subgraph passed as a raw ``GraphModule`` arg (which
        fails to compile). Fresh get_attr nodes (same target, same submodule)
        keep each flex region self-contained. Plain-tensor get_attrs are left
        shared: they are never annotated, so they stay cross-region inputs for
        both the original and the dup, exactly as before.
        """
        dups: dict[fx.Node, fx.Node] = {}
        for arg in fwd_node.all_input_nodes:
            if arg.op != "get_attr":
                continue
            if not isinstance(getattr(gm, arg.target, None), fx.GraphModule):
                continue
            with gm.graph.inserting_before(bwd_target):
                dup_attr = gm.graph.get_attr(arg.target)
                dup_attr.meta.update(arg.meta)  # shares arg's nested custom dict
                _privatize_custom_meta(dup_attr)
            dups[arg] = dup_attr
        return dups

    # Iterate the claimed must_recompute fwd nodes in graph order so that
    # each dup's upstream deps are already duped (and visible via
    # ``recomputed_nodes``) by the time we copy a downstream node.
    for fwd_node in sorted(remat_targets, key=order.__getitem__):
        bwd_target = remat_targets[fwd_node]
        # Pre-hoist offload reload chains for any args referencing offloaded
        # forward nodes, so the chain executes before the dup we're about to
        # create. Mirrors upstream's eager-copy-into-new-graph trick.
        for arg in fwd_node.all_input_nodes:
            bwd_wait = offloaded_fwd_to_bwd_wait.get(arg)
            if bwd_wait is not None:
                ensure_offload_chain_before(bwd_wait, bwd_target)
        # HOP subgraph get_attrs must be private to each recompute dup so
        # regional_inductor can partition each flex region independently.
        get_attr_dups = duplicate_subgraph_get_attrs(fwd_node, bwd_target)

        def remat_and_dup_get_attr(x: object, _dups=get_attr_dups) -> object:
            if isinstance(x, fx.Node) and x in _dups:
                return _dups[x]
            return remat_input(x)

        with gm.graph.inserting_before(bwd_target):
            dup = gm.graph.node_copy(fwd_node, remat_and_dup_get_attr)
            _privatize_custom_meta(dup)  # node_copy shares fwd_node's custom dict
        dup.name = fwd_node.name + "_recomputed"
        dup.meta["autograd_backward"] = True
        recomputed_nodes[fwd_node] = dup
        log.debug(
            "Recomputing %s before backward node %s", fwd_node.name, bwd_target.name
        )

    # Redirect every direct backward consumer of a recomputed forward node
    # to read from the dup. Backward consumers of offloaded forward nodes
    # were already redirected to their reload chain by the CPU offload
    # pass, so the offload branch of remat_input is a no-op here.
    direct_bwd_consumers = {
        user
        for fwd_node in recomputed_nodes
        for user in fwd_node.users
        if _is_backward_node(user)
    }
    for bwd_node in direct_bwd_consumers:
        bwd_node.args = torch.fx.map_arg(bwd_node.args, remat_input)
        bwd_node.kwargs = torch.fx.map_arg(bwd_node.kwargs, remat_input)

    # Targeted erase: original forward must_recompute nodes whose consumers
    # were all backward now have no users and can be removed. Originals with
    # remaining forward consumers stay in place. Iterate in reverse graph
    # order so downstream originals are erased first, freeing their upstream
    # originals' user lists for erase in the same pass.
    for orig in reversed(list(recomputed_nodes)):
        if not orig.users:
            log.debug(
                "erased %s, in replace of %s",
                orig.name,
                recomputed_nodes[orig].name,
            )
            gm.graph.erase_node(orig)

    # raise_getitems pass for better memory (like default_partition)
    gm = raise_getitems(gm)

    gm.recompile()
    return gm
