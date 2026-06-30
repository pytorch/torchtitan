# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import deepcopy

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.fx._lazy_graph_module import _make_graph_module

from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
    find_fsdp_unshard_output,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    allow_fx_graph_extraction_of_side_effectful_ops,
    graph_outputs,
    output_names,
    placeholder_names,
    trace_graph_pp_graph,
    unique_in_order,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPForwardSplit:
    """Forward graph split around FSDP unshard collectives.

    Attributes:
        unshard_module (fx.GraphModule | None): Graph that turns flat
            parameter inputs into unsharded parameter values, or ``None`` when
            the forward graph has no FSDP unshard collective.
        fw_no_fsdp_module (fx.GraphModule): Forward graph with FSDP unshard
            collectives removed.
        unshard_flat_param_indices (tuple[int, ...]): Flat parameter indices
            consumed by ``unshard_module``.
        unshard_output_names (tuple[str, ...]): ``unshard_module`` output
            names.
        fw_no_fsdp_input_names (tuple[str, ...]): ``fw_no_fsdp_module``
            placeholder names.
        fw_no_fsdp_flat_input_indices (tuple[int, ...]): Flat traced input
            indices for non-parameter inputs still consumed by
            ``fw_no_fsdp_module``.
        num_fw_unsharded_param_inputs (int): Number of leading
            ``fw_no_fsdp_module`` inputs supplied by ``unshard_module``.
        fw_no_fsdp_output_names (tuple[str, ...]): ``fw_no_fsdp_module`` output
            names.
    """

    unshard_module: fx.GraphModule | None
    fw_no_fsdp_module: fx.GraphModule
    unshard_flat_param_indices: tuple[int, ...]
    unshard_output_names: tuple[str, ...]
    fw_no_fsdp_input_names: tuple[str, ...]
    fw_no_fsdp_flat_input_indices: tuple[int, ...]
    num_fw_unsharded_param_inputs: int
    fw_no_fsdp_output_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPBackwardSplit:
    """Backward graph split around FSDP/DDP/HSDP reduce-grad collectives.

    Attributes:
        bw_no_fsdp_module (fx.GraphModule): Backward graph with reduce-grad
            epilogues removed from parameter-gradient outputs.
        reduce_grad_module (fx.GraphModule | None): Graph that performs
            reduce-scatter/all-reduce epilogues for parameter gradients, or
            ``None`` when no reduce-grad collective exists.
        bw_no_fsdp_output_names (tuple[str, ...]): ``bw_no_fsdp_module`` output
            names.
        reduce_grad_input_names (tuple[str, ...]): ``reduce_grad_module``
            placeholder names, or empty when ``reduce_grad_module`` is
            ``None``.
    """

    bw_no_fsdp_module: fx.GraphModule
    reduce_grad_module: fx.GraphModule | None
    bw_no_fsdp_output_names: tuple[str, ...]
    reduce_grad_input_names: tuple[str, ...]


def split_forward_fsdp_collectives(
    fw_module: fx.GraphModule,
    *,
    num_params: int,
    fwd_input_names: tuple[str, ...],
    fwd_flat_input_indices: tuple[int, ...],
) -> GraphPPFSDPForwardSplit:
    """Split forward FSDP all-gather chains from a forward graph.

    Contract:
      unshard(param_shards_and_replicated_params)
        -> unsharded_param_values

      fw_no_fsdp(unsharded_param_values, remaining_forward_inputs)
        -> original_forward_outputs

    Flat traced inputs are ordered as params, buffers, then user inputs. Any
    forward placeholder whose flat input index is less than ``num_params`` is a
    parameter input. Inputs with an all-gather/wait/view chain become
    unsharded values. Replicated or otherwise non-sharded params pass through
    the unshard graph so ``fw_no_fsdp`` still receives one value per original
    parameter input. If no all-gather chain exists, the split is a no-op.

    Args:
        fw_module (fx.GraphModule): Forward graph produced by GraphPP
            partitioning.
        num_params (int): Number of flat traced inputs that are parameters.
        fwd_input_names (tuple[str, ...]): Forward graph placeholder names from
            partition metadata.
        fwd_flat_input_indices (tuple[int, ...]): Flat traced input index for
            each forward graph placeholder.

    Returns:
        GraphPPFSDPForwardSplit: Forward split modules and calling-convention
        metadata.

    Raises:
        ValueError: If the provided forward input metadata does not match the
            graph placeholders.
    """
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")

    graph = deepcopy(fw_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    if len(fwd_input_names) != len(placeholders):
        raise ValueError(
            "Forward input names must match placeholder count: "
            f"{len(fwd_input_names)} != {len(placeholders)}"
        )
    if len(fwd_flat_input_indices) != len(placeholders):
        raise ValueError(
            "Forward flat input indices must match placeholder count: "
            f"{len(fwd_flat_input_indices)} != {len(placeholders)}"
        )
    if tuple(node.name for node in placeholders) != fwd_input_names:
        raise ValueError(
            "Forward input names must match graph placeholders: "
            f"expected {tuple(node.name for node in placeholders)}, "
            f"got {fwd_input_names}"
        )
    invalid_indices = sorted(index for index in fwd_flat_input_indices if index < 0)
    if invalid_indices:
        raise ValueError(
            "Forward flat input indices must be non-negative: " f"{invalid_indices}"
        )

    param_inputs: list[fx.Node] = []
    param_flat_indices: list[int] = []
    remaining_inputs: list[fx.Node] = []
    remaining_flat_input_indices: list[int] = []
    for node, flat_index in zip(placeholders, fwd_flat_input_indices, strict=True):
        if flat_index < num_params:
            param_inputs.append(node)
            param_flat_indices.append(flat_index)
        else:
            remaining_inputs.append(node)
            remaining_flat_input_indices.append(flat_index)

    unshard_outputs: list[object] = []
    found_collective = False

    for param_input in param_inputs:
        unshard_output = find_fsdp_unshard_output(param_input)
        if unshard_output is None:
            unshard_outputs.append(param_input)
            continue
        found_collective = True
        unshard_outputs.append(unshard_output)

    if not found_collective:
        trace_graph_pp_graph("graph_pp_fsdp_forward_no_fsdp", fw_module)
        return GraphPPFSDPForwardSplit(
            unshard_module=None,
            fw_no_fsdp_module=fw_module,
            unshard_flat_param_indices=(),
            unshard_output_names=(),
            fw_no_fsdp_input_names=fwd_input_names,
            fw_no_fsdp_flat_input_indices=fwd_flat_input_indices,
            num_fw_unsharded_param_inputs=0,
            fw_no_fsdp_output_names=output_names(fw_module),
        )

    all_outputs = graph_outputs(graph)
    output_node = graph.find_nodes(op="output")[0]
    graph_output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(all_outputs))
    )
    unshard_output_descs = [None] * len(unshard_outputs)

    with allow_fx_graph_extraction_of_side_effectful_ops(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        unshard_graph = _extract_graph_with_inputs_outputs(
            graph,
            param_inputs,
            unshard_outputs,
            unshard_output_descs,
            "unshard",
            ignore_must_be_in_fw_bw=True,
        )
        fw_no_fsdp_graph = _extract_graph_with_inputs_outputs(
            graph,
            unshard_outputs + remaining_inputs,
            list(all_outputs),
            graph_output_descs,
            "fw_no_fsdp",
            ignore_must_be_in_fw_bw=True,
        )

    unshard_module = _make_graph_module(fw_module, unshard_graph)
    fw_no_fsdp_module = _make_graph_module(fw_module, fw_no_fsdp_graph)
    trace_graph_pp_graph("graph_pp_fsdp_unshard", unshard_module)
    trace_graph_pp_graph("graph_pp_fsdp_forward_no_fsdp", fw_no_fsdp_module)
    unshard_output_names = output_names(unshard_module)
    return GraphPPFSDPForwardSplit(
        unshard_module=unshard_module,
        fw_no_fsdp_module=fw_no_fsdp_module,
        unshard_flat_param_indices=tuple(param_flat_indices),
        unshard_output_names=unshard_output_names,
        fw_no_fsdp_input_names=placeholder_names(fw_no_fsdp_module),
        fw_no_fsdp_flat_input_indices=tuple(remaining_flat_input_indices),
        num_fw_unsharded_param_inputs=len(unshard_output_names),
        fw_no_fsdp_output_names=output_names(fw_no_fsdp_module),
    )


def split_backward_fsdp_collectives(
    bw_module: fx.GraphModule,
    *,
    num_param_grads: int,
) -> GraphPPFSDPBackwardSplit:
    """Split backward FSDP/DDP/HSDP reduce-grad epilogues.

    Contract:
      bw_no_fsdp(original_backward_inputs)
        -> reduce_grad_inputs, remaining_backward_outputs

      reduce_grad(unique_reduce_grad_inputs)
        -> original_param_grad_outputs

    Backward outputs are ordered as parameter-grad leaves followed by input
    grads. Parameter-grad slots that do not end in a reduce-scatter/all-reduce
    chain, including ``None`` slots for unused or non-differentiable params,
    are kept in place to preserve the one-output-per-param-grad calling
    convention.

    NOTE: The pre-reduce dtype cast remains in ``bw_no_fsdp``. This matches
    eager FSDP accumulation with gradient sync disabled, where local grads are
    accumulated in the reduce dtype and reduced once later.

    Args:
        bw_module (fx.GraphModule): Backward graph produced by GraphPP
            partitioning.
        num_param_grads (int): Number of leading backward outputs that are
            parameter-gradient slots.

    Returns:
        GraphPPFSDPBackwardSplit: Backward split modules and
        calling-convention metadata.

    Raises:
        ValueError: If ``num_param_grads`` is invalid for the backward graph
            outputs.
    """
    if num_param_grads < 0:
        raise ValueError(f"num_param_grads must be non-negative, got {num_param_grads}")

    graph = deepcopy(bw_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    all_outputs = graph_outputs(graph)
    if num_param_grads > len(all_outputs):
        raise ValueError(
            "num_param_grads cannot exceed backward output count: "
            f"{num_param_grads} > {len(all_outputs)}"
        )
    grad_outputs = all_outputs[:num_param_grads]
    remaining_outputs = all_outputs[num_param_grads:]
    output_node = graph.find_nodes(op="output")[0]
    output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(all_outputs))
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
        trace_graph_pp_graph("graph_pp_fsdp_backward_no_fsdp", bw_module)
        return GraphPPFSDPBackwardSplit(
            bw_no_fsdp_module=bw_module,
            reduce_grad_module=None,
            bw_no_fsdp_output_names=output_names(bw_module),
            reduce_grad_input_names=(),
        )

    unique_reduce_grad_inputs = unique_in_order(
        input_node
        for input_node in reduce_grad_inputs
        if isinstance(input_node, fx.Node)
    )
    bw_no_fsdp_output_descs = [None] * len(reduce_grad_inputs)
    with allow_fx_graph_extraction_of_side_effectful_ops(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        bw_no_fsdp_graph = _extract_graph_with_inputs_outputs(
            graph,
            placeholders,
            reduce_grad_inputs + list(remaining_outputs),
            bw_no_fsdp_output_descs + remaining_output_descs,
            "bw_no_fsdp",
            ignore_must_be_in_fw_bw=True,
        )
        reduce_grad_graph = _extract_graph_with_inputs_outputs(
            graph,
            unique_reduce_grad_inputs,
            list(grad_outputs),
            grad_output_descs,
            "reduce_grad",
            ignore_must_be_in_fw_bw=True,
        )

    bw_no_fsdp_module = _make_graph_module(bw_module, bw_no_fsdp_graph)
    reduce_grad_module = _make_graph_module(bw_module, reduce_grad_graph)
    trace_graph_pp_graph("graph_pp_fsdp_backward_no_fsdp", bw_no_fsdp_module)
    trace_graph_pp_graph("graph_pp_fsdp_reduce_grad", reduce_grad_module)
    return GraphPPFSDPBackwardSplit(
        bw_no_fsdp_module=bw_no_fsdp_module,
        reduce_grad_module=reduce_grad_module,
        bw_no_fsdp_output_names=output_names(bw_no_fsdp_module),
        reduce_grad_input_names=placeholder_names(reduce_grad_module),
    )
