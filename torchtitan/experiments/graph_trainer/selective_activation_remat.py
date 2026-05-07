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
    cleanup_recompute_tags,
    force_save_bw_mutation_src,
    has_recomputable_ops,
    has_recomputable_rng_ops,
    must_recompute,
)


log = logging.getLogger(__name__)
_EMPTY_CUSTOM_META: dict[str, object] = {}


def _is_backward_node(node: fx.Node, use_phase: bool = False) -> bool:
    """Check if node is in backward region.

    If use_phase is True, only checks custom["phase"] == "backward"
    (user annotation). Otherwise falls back to node.meta["autograd_backward"],
    which Dynamo adds when tracing torch.autograd.grad.
    """
    custom = node.meta.get("custom", _EMPTY_CUSTOM_META)
    if use_phase:
        return custom.get("phase") == "backward"
    return node.meta.get("autograd_backward", False)


def _has_user_phase_annotation(gm: fx.GraphModule) -> bool:
    """Check if any node has the user-level phase: backward annotation."""
    return any(
        node.meta.get("custom", _EMPTY_CUSTOM_META).get("phase") == "backward"
        for node in gm.graph.nodes
    )


def _collect_backward_regions(
    gm: fx.GraphModule, use_phase: bool
) -> list[tuple[int, int, bool]]:
    """Returns (bwd_start, bwd_end, needs_remat) for each backward region.

    Regions are maximal contiguous runs of backward nodes, as [start, end)
    indices into the graph node list.
    """
    regions: list[tuple[int, int, bool]] = []
    bwd_start: int | None = None
    needs_remat = False

    for idx, node in enumerate(gm.graph.nodes):
        if _is_backward_node(node, use_phase=use_phase):
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
        regions.append((bwd_start, idx + 1, needs_remat))

    return regions


def remat_using_tags_for_fwd_loss_bwd_graph(
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

    Backward regions are identified by ``custom["phase"] == "backward"``
    (user annotation) or ``node.meta["autograd_backward"] == True`` (set by
    Dynamo when tracing ``torch.autograd.grad``). When the user provides
    phase annotations, only those annotated regions are used.

    The graph may contain multiple disjoint backward regions (e.g. chunked
    loss). Regions that do not depend on recomputable forward nodes are
    skipped. Only one region may require remat; if multiple do, we error
    and ask the user to annotate which region to rematerialize.
    """
    del example_inputs  # unused; accepted for graph pass signature compatibility

    if not has_recomputable_ops(gm):
        return gm

    if has_recomputable_rng_ops(gm):
        raise RuntimeError(
            "Activation checkpoint rematerialization in `forward-loss-backward` graph does not support RNG ops "
            "in recompute regions. Please move RNG operations outside "
            "of recompute regions, or use joint graph mode (where partitioner handles RNG)."
        )

    # Use partitioner pass to normalize AC node tags.
    gm = cleanup_recompute_tags(gm, is_default_partition=True)

    force_save_bw_mutation_src(gm)

    # must_recompute (used inside _collect_backward_regions) requires
    # cleanup_recompute_tags to have run first.
    use_phase = _has_user_phase_annotation(gm)
    regions = _collect_backward_regions(gm, use_phase)
    if not regions:
        return gm

    # User-annotated phase regions: multiple annotations is always an error.
    if use_phase and len(regions) > 1:
        raise RuntimeError(
            f"Detected {len(regions)} disjoint backward regions annotated with "
            'phase: "backward" but remat only supports a single backward region. '
            "Please ensure only one contiguous region is annotated."
        )

    remat_regions = [(s, e) for s, e, needs in regions if needs]

    if len(remat_regions) > 1:
        raise RuntimeError(
            f"Detected {len(remat_regions)} disjoint backward regions that require recomputation, "
            "but remat only supports one such region in a forward-loss-backward graph."
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

    def claim_for(bwd_node: fx.Node) -> None:
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
        claim_for(bwd_node)

    # Map original forward must_recompute node -> its recomputed duplicate.
    recomputed_nodes: dict[fx.Node, fx.Node] = {}
    # CPU offload: track which bwd target each reload-chain node was last
    # hoisted before, so we can re-hoist if an earlier dup needs it later.
    moved_offload: dict[fx.Node, fx.Node] = {}

    def ensure_offload_chain_before(reload_node: fx.Node, target: fx.Node) -> None:
        """Move ``reload_node`` and its bwd-region deps in front of ``target``.

        Mirrors upstream's ``_ensure_in_env``: a recompute dup consuming an
        offloaded forward node must read from the reload chain on GPU, not
        from F's freed storage. ``moved_offload`` keeps moves idempotent
        and ensures the chain ends up before the earliest target across
        repeated calls.
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
            prev = moved_offload.get(n)
            if prev is not None and order[prev] <= order[target]:
                continue
            bwd_reload_chain.add(n)
            stack.extend(n.all_input_nodes)

        for n in all_nodes:
            if n in bwd_reload_chain:
                target.prepend(n)
                moved_offload[n] = target

    def remat_input(x: object) -> object:
        """Arg-transform: redirect must_recompute originals to their dups, and
        offloaded forward nodes to their CPU-reload chain. Hoisting of the
        reload chain happens separately in the dup-creation loop."""
        if not isinstance(x, fx.Node):
            return x
        if x in recomputed_nodes:
            return recomputed_nodes[x]
        reload_node = x.meta.get("cpu_offload_reload_node")
        if reload_node is not None:
            return reload_node
        return x

    # Walk the existing graph in forward order. Each claimed must_recompute
    # node is duplicated immediately before its target backward node; topology
    # is satisfied by the graph's own order — upstream deps appear earlier in
    # the walk, so their duplicates are inserted (and visible via
    # ``recomputed_nodes``) before any downstream duplicate references them.
    for fwd_node in all_nodes[:bwd_start]:
        bwd_target = remat_targets.get(fwd_node)
        if bwd_target is None:
            continue
        # Pre-hoist offload reload chains for any args referencing offloaded
        # forward nodes, so the chain executes before the dup we're about to
        # create. Mirrors upstream's eager-copy-into-new-graph trick.
        for arg in fwd_node.all_input_nodes:
            reload_node = arg.meta.get("cpu_offload_reload_node")
            if reload_node is not None:
                ensure_offload_chain_before(reload_node, bwd_target)
        with gm.graph.inserting_before(bwd_target):
            dup = gm.graph.node_copy(fwd_node, remat_input)
        dup.name = fwd_node.name + "_recomputed"
        dup.meta["autograd_backward"] = True
        recomputed_nodes[fwd_node] = dup
        log.debug(
            "Recomputing %s before backward node %s", fwd_node.name, bwd_target.name
        )

    # Redirect each backward node's inputs to the recomputed duplicates.
    # Backward consumers of offloaded forward nodes were already redirected
    # to their reload chain by the CPU offload pass, so the offload branch
    # of remat_input is a no-op here.
    for bwd_node in bwd_nodes:
        bwd_node.args = torch.fx.map_arg(bwd_node.args, remat_input)
        bwd_node.kwargs = torch.fx.map_arg(bwd_node.kwargs, remat_input)

    # Targeted erase: original forward must_recompute nodes whose consumers
    # were all backward now have no users and can be removed. Originals with
    # remaining forward consumers stay in place. Iterate in reverse graph
    # order so downstream originals are erased first, freeing their upstream
    # originals' user lists for erase in the same pass.
    for orig in reversed(list(recomputed_nodes)):
        if not orig.users:
            gm.graph.erase_node(orig)

    # raise_getitems pass for better memory (like default_partition)
    gm = raise_getitems(gm)

    gm.graph.lint()
    return gm
