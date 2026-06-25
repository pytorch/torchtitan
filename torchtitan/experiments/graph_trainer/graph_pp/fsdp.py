# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import torch
import torch.fx as fx
import torch.fx.node
import torch.utils._pytree as pytree

from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
    find_fsdp_unshard_output,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import GraphPPInputSource
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    _graph_outputs,
    _make_graph_module_like,
    _output_names,
    _placeholder_names,
    extract_graph_with_graph_pp_abi,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPForwardSplit:
    unshard_module: fx.GraphModule | None
    fw_no_fsdp_module: fx.GraphModule
    unshard_input_sources: tuple[GraphPPInputSource, ...]
    unshard_output_names: tuple[str, ...]
    fw_no_fsdp_input_sources: tuple[GraphPPInputSource, ...]
    fw_no_fsdp_output_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPBackwardSplit:
    bw_no_fsdp_module: fx.GraphModule
    reduce_grad_module: fx.GraphModule | None
    bw_no_fsdp_output_names: tuple[str, ...]
    reduce_grad_input_names: tuple[str, ...]


@contextmanager
def _exclude_from_fx_side_effectful(exclude_vals: set[Any]):
    # NOTE: FX does not expose a scoped public API for treating wait_tensor as
    # extractable during graph splitting, so GraphPP restores this private set
    # immediately after each extraction.
    original_val = torch.fx.node._side_effectful_functions.copy()
    try:
        torch.fx.node._side_effectful_functions -= exclude_vals
        yield
    finally:
        torch.fx.node._side_effectful_functions.clear()
        torch.fx.node._side_effectful_functions.update(original_val)


def split_forward_fsdp_collectives(
    fw_module: fx.GraphModule,
    *,
    num_params: int,
    fwd_input_sources: tuple[GraphPPInputSource, ...],
) -> GraphPPFSDPForwardSplit:
    """Split forward FSDP all-gather chains from a forward graph.

    Contract:
      Parameter placeholders are the leading flat state inputs from the stage
      trace. If an input feeds an all-gather/wait/view chain, GraphPP extracts
      that chain into an ``unshard`` graph and rewrites the no-FSDP forward
      graph to consume the unsharded values. If no all-gather chain exists, the
      split is a no-op and the original forward graph is returned.

    Pseudocode:
      find parameter placeholders from GraphPPInputSource metadata
      for each param placeholder, find the FSDP unshard output node
      extract unshard(param shards) -> unsharded params
      extract fw_no_fsdp(unsharded params, remaining inputs) -> original fw outputs
      return both graph variants plus updated input-source metadata

    NOTE: FSDP region detection uses the shared chain helper in
    ``fsdp_patterns.py``. It intentionally matches the default memory policy's
    force-save node for ``reshard_after_forward=False``.
    """
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")

    graph = deepcopy(fw_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    if len(placeholders) != len(fwd_input_sources):
        raise ValueError(
            "Forward input source metadata must match placeholder count: "
            f"{len(fwd_input_sources)} != {len(placeholders)}"
        )
    param_inputs = [
        node
        for node, source in zip(placeholders, fwd_input_sources, strict=True)
        if source.kind == "flat_input" and source.index < num_params
    ]
    remaining_inputs = [
        node
        for node, source in zip(placeholders, fwd_input_sources, strict=True)
        if not (source.kind == "flat_input" and source.index < num_params)
    ]
    remaining_input_sources = [
        source
        for source in fwd_input_sources
        if not (source.kind == "flat_input" and source.index < num_params)
    ]
    unshard_outputs: list[Any] = []
    found_collective = False

    for param_input in param_inputs:
        unshard_output = find_fsdp_unshard_output(param_input)
        if unshard_output is None:
            unshard_outputs.append(param_input)
            continue
        found_collective = True
        unshard_outputs.append(unshard_output)

    if not found_collective:
        return GraphPPFSDPForwardSplit(
            unshard_module=None,
            fw_no_fsdp_module=fw_module,
            unshard_input_sources=(),
            unshard_output_names=(),
            fw_no_fsdp_input_sources=fwd_input_sources,
            fw_no_fsdp_output_names=_output_names(fw_module),
        )

    graph_outputs = _graph_outputs(graph)
    output_node = graph.find_nodes(op="output")[0]
    graph_output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(graph_outputs))
    )
    unshard_output_descs = [None] * len(unshard_outputs)

    with _exclude_from_fx_side_effectful(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        unshard_graph = extract_graph_with_graph_pp_abi(
            graph,
            param_inputs,
            unshard_outputs,
            unshard_output_descs,
        )
        fw_no_fsdp_graph = extract_graph_with_graph_pp_abi(
            graph,
            unshard_outputs + remaining_inputs,
            list(graph_outputs),
            graph_output_descs,
        )

    unshard_module = _make_graph_module_like(fw_module, unshard_graph)
    fw_no_fsdp_module = _make_graph_module_like(fw_module, fw_no_fsdp_graph)
    unshard_output_names = _output_names(unshard_module)
    unshard_input_sources = tuple(
        GraphPPInputSource(
            name=name,
            kind="unsharded_param",
            index=index,
        )
        for index, name in enumerate(unshard_output_names)
    )
    return GraphPPFSDPForwardSplit(
        unshard_module=unshard_module,
        fw_no_fsdp_module=fw_no_fsdp_module,
        unshard_input_sources=tuple(
            source
            for source in fwd_input_sources
            if source.kind == "flat_input" and source.index < num_params
        ),
        unshard_output_names=unshard_output_names,
        fw_no_fsdp_input_sources=unshard_input_sources + tuple(remaining_input_sources),
        fw_no_fsdp_output_names=_output_names(fw_no_fsdp_module),
    )


def split_backward_fsdp_collectives(
    bw_module: fx.GraphModule,
    *,
    num_param_grads: int,
) -> GraphPPFSDPBackwardSplit:
    """Split backward FSDP reduce-scatter epilogues from a backward graph.

    Contract:
      Backward outputs are ordered as parameter-grad leaves followed by input
      grads. If a parameter grad output ends in a reduce-scatter chain, GraphPP
      extracts that chain into a ``reduce_grad`` graph. The no-FSDP backward
      graph returns unreduced grad leaves, allowing GraphPP to reduce once at
      the end of the PP step. If no reduce-scatter exists, the split is a no-op.

    Pseudocode:
      split backward outputs into param grads and remaining outputs
      walk each param grad backward to the FSDP reduce-scatter input
      extract bw_no_fsdp(original inputs) -> unreduced grads + remaining outputs
      extract reduce_grad(unreduced grads) -> reduced param grads
      return both graphs plus the reduce-grad input names

    NOTE: The pre-reduce dtype cast remains in ``bw_no_fsdp``. This matches
    eager FSDP accumulation with gradient sync disabled, where local grads are
    accumulated in the reduce dtype and reduced once later.
    """
    if num_param_grads < 0:
        raise ValueError(f"num_param_grads must be non-negative, got {num_param_grads}")

    graph = deepcopy(bw_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    graph_outputs = _graph_outputs(graph)
    grad_outputs = graph_outputs[:num_param_grads]
    remaining_outputs = graph_outputs[num_param_grads:]
    output_node = graph.find_nodes(op="output")[0]
    output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(graph_outputs))
    )
    grad_output_descs = output_descs[:num_param_grads]
    remaining_output_descs = output_descs[num_param_grads:]

    reduce_grad_inputs = []
    found_collective = False
    for grad_output in grad_outputs:
        reduce_grad_input = find_fsdp_reduce_grad_input(grad_output)
        if reduce_grad_input is not None:
            found_collective = True
            reduce_grad_inputs.append(reduce_grad_input)
        else:
            reduce_grad_inputs.append(grad_output)

    if not found_collective:
        return GraphPPFSDPBackwardSplit(
            bw_no_fsdp_module=bw_module,
            reduce_grad_module=None,
            bw_no_fsdp_output_names=_output_names(bw_module),
            reduce_grad_input_names=(),
        )

    unique_reduce_grad_inputs = list(dict.fromkeys(reduce_grad_inputs))
    bw_no_fsdp_output_descs = [None] * len(reduce_grad_inputs)
    reduce_grad_input_descs = [None] * len(unique_reduce_grad_inputs)
    with _exclude_from_fx_side_effectful(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        bw_no_fsdp_graph = extract_graph_with_graph_pp_abi(
            graph,
            placeholders,
            reduce_grad_inputs + list(remaining_outputs),
            bw_no_fsdp_output_descs + remaining_output_descs,
        )
        reduce_grad_graph = extract_graph_with_graph_pp_abi(
            graph,
            unique_reduce_grad_inputs,
            list(grad_outputs),
            grad_output_descs,
        )

    bw_no_fsdp_module = _make_graph_module_like(bw_module, bw_no_fsdp_graph)
    reduce_grad_module = _make_graph_module_like(bw_module, reduce_grad_graph)
    return GraphPPFSDPBackwardSplit(
        bw_no_fsdp_module=bw_no_fsdp_module,
        reduce_grad_module=reduce_grad_module,
        bw_no_fsdp_output_names=_output_names(bw_no_fsdp_module),
        reduce_grad_input_names=_placeholder_names(reduce_grad_module),
    )
