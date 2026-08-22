# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in EP scheduling that defers dW-only backward work into A2A windows.

Contract
========
This pass composes the reusable dI/dW classification from
``backward_partition`` with the ``ep_overlap_pass`` scheduler. It consumes the
same chunked-region metadata as ``ep_overlap_schedule_pass`` and adds one
scheduling preference: backward weight-gradient work that provably feeds only
parameter-gradient outputs is withheld until every backward token-exchange
launch of its region is in flight, then emitted between those launches and
their waits. Gradient reductions and other collectives are never part of the
deferred set and keep their baseline order. Baseline scheduling contract
violations still raise; a graph with no legally deferrable dW work keeps the
baseline schedule with a debug message rather than an error.

Per-region dI/dW boundary
=========================
The traced train step returns ``[loss] + param_grads`` (``make_fwd_bwd_step``),
so the whole-step graph has no input-gradient outputs and a whole-graph di/dw
split would classify every backward node as dW-only. The deferral boundary is
therefore anchored per backward region:

* ``input_grad_outputs`` are the backward EP token-exchange launches
  themselves. Every dgrad value must reach the next all-to-all (this region's
  dispatch backward, or an earlier layer's backward exchange downstream), so
  ancestors of any backward launch are latency-critical and never deferred.
* ``param_grad_outputs`` are the parameter-gradient graph outputs (the flat
  outputs after the leading non-gradient entries). Their ancestors inside a
  backward chunk body form the wgrad chain.

``movable_nodes`` from ``partition_backward_nodes`` intersected with the
backward chunk bodies is then exactly the weight-gradient-only work
(grouped-mm wgrads plus their quantize/transpose plumbing) that may move into
a launch -> wait window. Classification is pure dataflow, so BF16 and
quantized backwards with the same dependency structure defer identically.
"""

from __future__ import annotations

from typing import Any

import torch.fx as fx

from torchtitan.experiments.graph_trainer.backward_partition import (
    partition_backward_nodes,
)
from torchtitan.experiments.graph_trainer.ep_overlap_pass import (
    _is_token_exchange_launch,
    ep_overlap_schedule_pass,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import collect_chunked_regions
from torchtitan.tools.logging import logger


def _param_grad_outputs(
    gm: fx.GraphModule, *, num_non_grad_outputs: int
) -> list[fx.Node]:
    """Return the parameter-gradient value nodes of the flat graph outputs.

    The traced train step returns ``[loss] + param_grads``, so parameter
    gradients are the flat outputs after the leading ``num_non_grad_outputs``
    entries. ``None`` grad slots and literal outputs are ignored.
    """
    output = next(node for node in gm.graph.nodes if node.op == "output")
    flat_outputs = output.args[0]
    if not isinstance(flat_outputs, (list, tuple)):
        flat_outputs = (flat_outputs,)
    return [
        value
        for value in flat_outputs[num_non_grad_outputs:]
        if isinstance(value, fx.Node)
    ]


def _deferred_dw_nodes(
    gm: fx.GraphModule,
    *,
    module_pattern: str,
    num_non_grad_outputs: int,
) -> set[fx.Node]:
    """Classify backward chunk bodies and return their deferrable dW set.

    Returns the empty set (with a debug message, never an error) when the
    graph has no backward token-exchange launches, no identifiable parameter
    gradients, or no separable movable dW-only nodes; the caller then keeps
    the baseline ep_overlap schedule.
    """
    backward_body_nodes = {
        node
        for region in collect_chunked_regions(gm, module_pattern=module_pattern)
        if region.is_backward
        for body in region.bodies_by_chunk.values()
        for node in body.nodes
    }
    backward_launches = [
        node
        for node in gm.graph.nodes
        if node in backward_body_nodes and _is_token_exchange_launch(node)
    ]
    if not backward_launches:
        logger.debug(
            "defer_param_grad found no backward EP token-exchange launches for "
            "pattern %s; keeping the baseline ep_overlap schedule.",
            module_pattern,
        )
        return set()
    param_grad_outputs = _param_grad_outputs(
        gm, num_non_grad_outputs=num_non_grad_outputs
    )
    if not param_grad_outputs:
        logger.debug(
            "defer_param_grad found no parameter-gradient graph outputs; "
            "keeping the baseline ep_overlap schedule.",
        )
        return set()
    partition = partition_backward_nodes(
        gm,
        input_grad_outputs=backward_launches,
        param_grad_outputs=param_grad_outputs,
    )
    deferred = partition.movable_nodes & backward_body_nodes
    if not deferred:
        logger.debug(
            "defer_param_grad found no separable movable dW-only nodes for "
            "pattern %s; keeping the baseline ep_overlap schedule.",
            module_pattern,
        )
    return deferred


def defer_param_grad_schedule_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = True,
    pair_first_token_exchange: bool = False,
    num_non_grad_outputs: int = 1,
) -> fx.GraphModule:
    """Reorder chunked EP regions with dW-only work deferred into A2A windows."""
    deferred = _deferred_dw_nodes(
        gm,
        module_pattern=module_pattern,
        num_non_grad_outputs=num_non_grad_outputs,
    )
    logger.info(
        "defer_param_grad classified %d deferrable dW node(s): module=%s",
        len(deferred),
        module_pattern,
    )
    return ep_overlap_schedule_pass(
        gm,
        example_inputs,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
        pair_first_token_exchange=pair_first_token_exchange,
        deferred_compute_nodes=deferred,
    )
