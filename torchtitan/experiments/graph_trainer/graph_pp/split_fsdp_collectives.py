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
    find_fsdp_unshard_outputs_by_param,
    is_all_gather_into_tensor,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    allow_fx_graph_extraction_of_side_effectful_ops,
    graph_outputs,
    output_names,
    placeholder_names,
    trace_graph_pp_graph,
    unique_in_order,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_MESH_AXIS_NAMES_META


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPUnshardExtraction:
    """Graph extraction result for FSDP unshard collectives.

    Attributes:
        unshard_module (fx.GraphModule | None): Graph that turns flat
            parameter inputs into unsharded parameter values, or ``None`` when
            the input graph has no FSDP unshard collective.
        compute_module (fx.GraphModule): Input graph with FSDP unshard
            collectives removed.
        unshard_flat_param_indices (tuple[int, ...]): Flat parameter indices
            consumed by ``unshard_module``.
        unshard_output_names (tuple[str, ...]): ``unshard_module`` output
            names.
        compute_input_names (tuple[str, ...]): ``compute_module``
            placeholder names.
        compute_flat_input_indices (tuple[int, ...]): Flat traced input indices
            for non-parameter inputs still consumed by ``compute_module``.
        num_compute_param_inputs (int): Number of leading ``compute_module``
            inputs supplied by ``unshard_module``.
        compute_output_names (tuple[str, ...]): ``compute_module`` output
            names.
    """

    unshard_module: fx.GraphModule | None
    compute_module: fx.GraphModule
    unshard_flat_param_indices: tuple[int, ...]
    unshard_output_names: tuple[str, ...]
    compute_input_names: tuple[str, ...]
    compute_flat_input_indices: tuple[int, ...]
    num_compute_param_inputs: int
    compute_output_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPReduceGradExtraction:
    """Graph extraction result for FSDP/DDP/HSDP gradient reduction.

    Attributes:
        compute_module (fx.GraphModule): Input graph with reduce-grad
            epilogues removed from parameter-gradient outputs.
        reduce_grad_module (fx.GraphModule | None): Graph that performs
            reduce-scatter/all-reduce epilogues for parameter gradients, or
            ``None`` when no reduce-grad collective exists.
        compute_output_names (tuple[str, ...]): ``compute_module`` output
            names.
        reduce_grad_input_names (tuple[str, ...]): ``reduce_grad_module``
            placeholder names, or empty when ``reduce_grad_module`` is
            ``None``.
        reduction_node_names (frozenset[str]): Nodes owned by the reduction
            epilogue, including when extraction is disabled.
    """

    compute_module: fx.GraphModule
    reduce_grad_module: fx.GraphModule | None
    compute_output_names: tuple[str, ...]
    reduce_grad_input_names: tuple[str, ...]
    reduction_node_names: frozenset[str] = frozenset()


def _is_expert_fsdp_node(node: object) -> bool:
    return isinstance(node, fx.Node) and "efsdp" in node.meta.get("custom", {}).get(
        FSDP_MESH_AXIS_NAMES_META, ()
    )


def _is_expert_fsdp_reduce_grad(
    grad_output: object,
    reduce_grad_input: fx.Node,
) -> bool:
    """Return whether a reduce-grad suffix belongs to the eFSDP mesh axis."""
    node = grad_output
    while isinstance(node, fx.Node):
        if _is_expert_fsdp_node(node):
            return True
        if node is reduce_grad_input or len(node.all_input_nodes) != 1:
            return False
        node = node.all_input_nodes[0]
    return False


def remove_fsdp_reduction_tail(
    fw_module: fx.GraphModule,
    *,
    reduction_node_names: frozenset[str],
) -> None:
    """Remove backward FSDP reduction nodes copied into the forward graph."""
    if not reduction_node_names:
        return
    # FX preserves mutation-only tails during DCE. The backward graph owns
    # these reduction nodes, so remove dead forward copies before recompiling.
    for node in reversed(list(fw_module.graph.nodes)):
        if node.name in reduction_node_names and not node.users:
            fw_module.graph.erase_node(node)
    fw_module.graph.lint()
    fw_module.recompile()


def _remove_dead_all_gather_launches(graph: fx.Graph) -> None:
    """Remove all-gather branches whose waits were excluded from a subgraph."""
    removable_inputs = set()
    for node in reversed(list(graph.nodes)):
        is_all_gather = is_all_gather_into_tensor(node) or (
            node.op == "call_function"
            and node.target
            == torch.ops._c10d_functional.all_gather_into_tensor_out.default
        )
        if not is_all_gather or node.users:
            continue
        pending = list(node.all_input_nodes)
        while pending:
            input_node = pending.pop()
            if input_node.op == "placeholder" or input_node in removable_inputs:
                continue
            removable_inputs.add(input_node)
            pending.extend(input_node.all_input_nodes)
        graph.erase_node(node)
    for node in reversed(list(graph.nodes)):
        if node in removable_inputs and not node.users:
            graph.erase_node(node)


def _parameter_inputs_needed_after_boundary(
    outputs: tuple[object, ...],
    *,
    boundary_outputs: list[object],
    param_inputs: list[fx.Node],
) -> list[fx.Node]:
    """Find parameter inputs reached without crossing an extracted boundary."""

    boundary_nodes = {
        output for output in boundary_outputs if isinstance(output, fx.Node)
    }
    needed_placeholders: set[fx.Node] = set()
    pending = [output for output in outputs if isinstance(output, fx.Node)]
    visited: set[fx.Node] = set()
    while pending:
        node = pending.pop()
        if node in visited or node in boundary_nodes:
            continue
        visited.add(node)
        if node.op == "placeholder":
            needed_placeholders.add(node)
            continue
        pending.extend(node.all_input_nodes)

    return [
        param_input
        for param_input in param_inputs
        if param_input in needed_placeholders and param_input not in boundary_nodes
    ]


def extract_fsdp_unshard_graph(
    graph_module: fx.GraphModule,
    *,
    num_params: int,
    input_names: tuple[str, ...],
    flat_input_indices: tuple[int, ...],
    side_effect_output_names: tuple[str, ...] = (),
    extract_fsdp_param_unshard: bool = True,
) -> GraphPPFSDPUnshardExtraction:
    """Extract FSDP parameter all-gather chains from a graph.

    Contract:
      unshard(param_shards_and_replicated_params)
        -> unsharded_param_values

      compute(unsharded_param_values, preserved_param_shards, remaining_inputs)
        -> original_outputs

    Flat traced inputs are ordered as params, buffers, then user inputs. Any
    placeholder whose flat input index is less than ``num_params`` is a
    parameter input. Inputs with an all-gather/wait/view chain become unsharded
    values. Replicated or otherwise non-sharded params pass through the unshard
    graph so ``compute`` still receives one value per original parameter input.
    Parameter shards still needed beyond the extracted boundary also pass
    through. This supports both forward-only and joint forward/backward graphs.
    If no all-gather chain exists, extraction is a no-op.

    Args:
        graph_module (fx.GraphModule): Graph containing FSDP unshard chains.
        num_params (int): Number of flat traced inputs that are parameters.
        input_names (tuple[str, ...]): Input graph placeholder names.
        flat_input_indices (tuple[int, ...]): Flat traced input index for each
            graph placeholder.
        side_effect_output_names (tuple[str, ...]): Mutation outputs that may
            move into the unshard graph.
        extract_fsdp_param_unshard (bool): Whether to extract the unshard graph.

    Returns:
        GraphPPFSDPUnshardExtraction: Extracted modules and calling-convention
            metadata.

    Raises:
        ValueError: If the provided input metadata does not match the graph
            placeholders.
    """
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")

    graph = deepcopy(graph_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    if len(input_names) != len(placeholders):
        raise ValueError(
            "Input names must match placeholder count: "
            f"{len(input_names)} != {len(placeholders)}"
        )
    if len(flat_input_indices) != len(placeholders):
        raise ValueError(
            "Flat input indices must match placeholder count: "
            f"{len(flat_input_indices)} != {len(placeholders)}"
        )
    if tuple(node.name for node in placeholders) != input_names:
        raise ValueError(
            "Input names must match graph placeholders: "
            f"expected {tuple(node.name for node in placeholders)}, "
            f"got {input_names}"
        )
    invalid_indices = sorted(index for index in flat_input_indices if index < 0)
    if invalid_indices:
        raise ValueError(
            "Flat input indices must be non-negative: " f"{invalid_indices}"
        )
    if not extract_fsdp_param_unshard:
        return GraphPPFSDPUnshardExtraction(
            unshard_module=None,
            compute_module=graph_module,
            unshard_flat_param_indices=(),
            unshard_output_names=(),
            compute_input_names=input_names,
            compute_flat_input_indices=flat_input_indices,
            num_compute_param_inputs=0,
            compute_output_names=output_names(graph_module),
        )

    param_inputs: list[fx.Node] = []
    param_flat_indices: list[int] = []
    remaining_inputs: list[fx.Node] = []
    remaining_flat_input_indices: list[int] = []
    for node, flat_index in zip(placeholders, flat_input_indices, strict=True):
        if flat_index < num_params:
            param_inputs.append(node)
            param_flat_indices.append(flat_index)
        else:
            remaining_inputs.append(node)
            remaining_flat_input_indices.append(flat_index)

    unshard_outputs: list[object] = []
    found_collective = False
    outputs_by_param = find_fsdp_unshard_outputs_by_param(param_inputs)

    for param_input in param_inputs:
        param_unshard_outputs = outputs_by_param[param_input]
        if not param_unshard_outputs:
            unshard_outputs.append(param_input)
            continue
        if len(param_unshard_outputs) != 1:
            raise ValueError(
                "GraphPP FSDP extraction expects one unshard chain per flat "
                f"parameter placeholder after deduplication, but "
                f"{param_input.name} has {len(param_unshard_outputs)}. "
                "Run deduplicate_fsdp_unshard_chains_pass before extraction."
            )
        unshard_output = param_unshard_outputs[0]
        if _is_expert_fsdp_node(unshard_output):
            unshard_outputs.append(param_input)
            continue
        found_collective = True
        unshard_outputs.append(unshard_output)

    if not found_collective:
        trace_graph_pp_graph("graph_pp_fsdp_compute_no_unshard", graph_module)
        return GraphPPFSDPUnshardExtraction(
            unshard_module=None,
            compute_module=graph_module,
            unshard_flat_param_indices=(),
            unshard_output_names=(),
            compute_input_names=input_names,
            compute_flat_input_indices=flat_input_indices,
            num_compute_param_inputs=0,
            compute_output_names=output_names(graph_module),
        )

    all_outputs = graph_outputs(graph)
    passthrough_param_inputs = _parameter_inputs_needed_after_boundary(
        all_outputs,
        boundary_outputs=unshard_outputs,
        param_inputs=param_inputs,
    )
    compute_param_inputs = [*unshard_outputs, *passthrough_param_inputs]
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
            compute_param_inputs,
            [*unshard_output_descs, *([None] * len(passthrough_param_inputs))],
            "unshard",
            ignore_must_be_in_fw_bw=True,
        )
        unshard_node_names = {node.name for node in unshard_graph.nodes}
        compute_outputs_and_descs = [
            (output, desc)
            for output, desc in zip(all_outputs, graph_output_descs, strict=True)
            if not (
                isinstance(output, fx.Node)
                and output.name in side_effect_output_names
                and output.name in unshard_node_names
            )
        ]
        compute_graph = _extract_graph_with_inputs_outputs(
            graph,
            compute_param_inputs + remaining_inputs,
            [output for output, _ in compute_outputs_and_descs],
            [desc for _, desc in compute_outputs_and_descs],
            "compute_no_unshard",
            ignore_must_be_in_fw_bw=True,
        )
        compute_graph.eliminate_dead_code(
            is_impure_node=lambda node: not (
                node.op == "call_function" and node.name in unshard_node_names
            )
            and node.is_impure()
        )

    # Extraction preserves mutation-only backward prefetch launches. They have
    # no wait in the unshard graph and must not run as part of UNSHARD.
    _remove_dead_all_gather_launches(unshard_graph)
    unshard_graph.lint()
    unshard_module = _make_graph_module(graph_module, unshard_graph)
    compute_module = _make_graph_module(graph_module, compute_graph)
    trace_graph_pp_graph("graph_pp_fsdp_unshard", unshard_module)
    trace_graph_pp_graph("graph_pp_fsdp_compute_no_unshard", compute_module)
    unshard_output_names = output_names(unshard_module)
    return GraphPPFSDPUnshardExtraction(
        unshard_module=unshard_module,
        compute_module=compute_module,
        unshard_flat_param_indices=tuple(param_flat_indices),
        unshard_output_names=unshard_output_names,
        compute_input_names=placeholder_names(compute_module),
        compute_flat_input_indices=tuple(remaining_flat_input_indices),
        num_compute_param_inputs=len(unshard_output_names),
        compute_output_names=output_names(compute_module),
    )


def extract_fsdp_reduce_grad_graph(
    graph_module: fx.GraphModule,
    *,
    num_param_grads: int,
    param_grad_output_start: int = 0,
    extract_grad_reduction: bool = True,
) -> GraphPPFSDPReduceGradExtraction:
    """Extract FSDP/DDP/HSDP reduce-grad epilogues from a graph.

    Contract:
      compute(original_inputs)
        -> leading_outputs, reduce_grad_inputs, trailing_outputs

      reduce_grad(unique_reduce_grad_inputs)
        -> original_param_grad_outputs

    Parameter-gradient outputs begin at ``param_grad_output_start``, permitting
    both backward-only graphs and joint graphs with leading outputs such as
    loss. Parameter-gradient slots that do not end in a
    reduce-scatter/all-reduce chain, including ``None`` slots for unused or
    non-differentiable params, are kept in place to preserve the
    one-output-per-param-grad calling convention.

    NOTE: The pre-reduce dtype cast remains in ``compute``. This matches
    eager FSDP accumulation with gradient sync disabled, where local grads are
    accumulated in the reduce dtype and reduced once later.

    Args:
        graph_module (fx.GraphModule): Graph containing parameter-gradient
            outputs.
        num_param_grads (int): Number of parameter-gradient output slots.
        param_grad_output_start (int): Index of the first parameter-gradient
            output. Defaults to zero for backward-only graphs.
        extract_grad_reduction (bool): Whether to extract the reduction graph.

    Returns:
        GraphPPFSDPReduceGradExtraction: Extracted modules and
            calling-convention metadata.

    Raises:
        ValueError: If the parameter-gradient output range is invalid.
    """
    if num_param_grads < 0:
        raise ValueError(f"num_param_grads must be non-negative, got {num_param_grads}")
    if param_grad_output_start < 0:
        raise ValueError(
            "param_grad_output_start must be non-negative, got "
            f"{param_grad_output_start}"
        )

    graph = deepcopy(graph_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    all_outputs = graph_outputs(graph)
    param_grad_output_end = param_grad_output_start + num_param_grads
    if param_grad_output_end > len(all_outputs):
        if param_grad_output_start == 0:
            raise ValueError(
                "num_param_grads cannot exceed backward output count: "
                f"{num_param_grads} > {len(all_outputs)}"
            )
        raise ValueError(
            "Parameter-gradient output range exceeds graph output count: "
            f"[{param_grad_output_start}, {param_grad_output_end}) with "
            f"{len(all_outputs)} outputs"
        )
    leading_outputs = all_outputs[:param_grad_output_start]
    grad_outputs = all_outputs[param_grad_output_start:param_grad_output_end]
    trailing_outputs = all_outputs[param_grad_output_end:]
    output_node = graph.find_nodes(op="output")[0]
    output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(all_outputs))
    )
    leading_output_descs = output_descs[:param_grad_output_start]
    grad_output_descs = output_descs[param_grad_output_start:param_grad_output_end]
    trailing_output_descs = output_descs[param_grad_output_end:]

    reduce_grad_inputs = []
    reduction_outputs = []
    found_collective = False
    for grad_output in grad_outputs:
        reduce_grad_input = find_fsdp_reduce_grad_input(grad_output)
        if reduce_grad_input is not None and not _is_expert_fsdp_reduce_grad(
            grad_output, reduce_grad_input
        ):
            found_collective = True
            reduction_outputs.append((grad_output, frozenset((reduce_grad_input,))))
            reduce_grad_inputs.append(reduce_grad_input)
        else:
            reduce_grad_inputs.append(grad_output)

    if not found_collective:
        trace_graph_pp_graph("graph_pp_fsdp_compute_no_reduce_grad", graph_module)
        return GraphPPFSDPReduceGradExtraction(
            compute_module=graph_module,
            reduce_grad_module=None,
            compute_output_names=output_names(graph_module),
            reduce_grad_input_names=(),
        )

    reduction_node_names = set()
    for grad_output, boundaries in reduction_outputs:
        pending = [grad_output]
        while pending:
            node = pending.pop()
            if not isinstance(node, fx.Node) or node in boundaries:
                continue
            if node.name in reduction_node_names:
                continue
            reduction_node_names.add(node.name)
            pending.extend(node.all_input_nodes)

    if not extract_grad_reduction:
        return GraphPPFSDPReduceGradExtraction(
            compute_module=graph_module,
            reduce_grad_module=None,
            compute_output_names=output_names(graph_module),
            reduce_grad_input_names=(),
            reduction_node_names=frozenset(reduction_node_names),
        )

    _remove_dead_all_gather_launches(graph)
    graph.eliminate_dead_code()
    graph.lint()
    unique_reduce_grad_inputs = unique_in_order(
        input_node
        for input_node in reduce_grad_inputs
        if isinstance(input_node, fx.Node)
    )
    compute_grad_output_descs = [None] * len(reduce_grad_inputs)
    with allow_fx_graph_extraction_of_side_effectful_ops(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        compute_graph = _extract_graph_with_inputs_outputs(
            graph,
            placeholders,
            [*leading_outputs, *reduce_grad_inputs, *trailing_outputs],
            [
                *leading_output_descs,
                *compute_grad_output_descs,
                *trailing_output_descs,
            ],
            "compute_no_reduce_grad",
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

    # FX preserves mutation-only tails during DCE. Remove the reduction tail
    # after its inputs become explicit outputs of the backward graph.
    for node in reversed(list(compute_graph.nodes)):
        if node.name in reduction_node_names and not node.users:
            compute_graph.erase_node(node)
    _remove_dead_all_gather_launches(compute_graph)
    compute_graph.lint()

    compute_module = _make_graph_module(graph_module, compute_graph)
    reduce_grad_module = _make_graph_module(graph_module, reduce_grad_graph)
    trace_graph_pp_graph("graph_pp_fsdp_compute_no_reduce_grad", compute_module)
    trace_graph_pp_graph("graph_pp_fsdp_reduce_grad", reduce_grad_module)
    return GraphPPFSDPReduceGradExtraction(
        compute_module=compute_module,
        reduce_grad_module=reduce_grad_module,
        compute_output_names=output_names(compute_module),
        reduce_grad_input_names=placeholder_names(reduce_grad_module),
        reduction_node_names=frozenset(reduction_node_names),
    )
