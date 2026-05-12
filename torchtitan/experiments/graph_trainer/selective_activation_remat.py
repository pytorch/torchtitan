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

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node


log = logging.getLogger(__name__)


def _collect_backward_regions(
    gm: fx.GraphModule,
) -> list[tuple[int, int, bool]]:
    """Returns (bwd_start, bwd_end, needs_remat) for each backward region.

    Regions are maximal contiguous runs of backward nodes, as [start, end)
    indices into the graph node list. This is still kind of OK for chunked
    loss because: (1) we would have errored earlier if there were multiple
    regions that need recompute, and (2) we only ever do recompute on the
    last backward.
    """
    regions: list[tuple[int, int, bool]] = []
    bwd_start: int | None = None
    needs_remat = False

    for idx, node in enumerate(gm.graph.nodes):
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
        regions.append((bwd_start, idx + 1, needs_remat))

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

    The graph may contain multiple disjoint backward regions (e.g. chunked
    loss). Regions that do not depend on recomputable forward nodes are
    skipped. Only one region may require remat; if multiple do, we error.
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

    # Assumption: chunked-loss regions (e.g. lm_head) do not carry AC, so
    # at most one backward region depends on must_recompute forward nodes.
    # If apply_sac_pass starts tagging the lm_head layer with AC, multiple
    # disjoint backward regions could need remat and this heuristic must
    # be revisited.
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
        with gm.graph.inserting_before(bwd_target):
            dup = gm.graph.node_copy(fwd_node, remat_input)
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
