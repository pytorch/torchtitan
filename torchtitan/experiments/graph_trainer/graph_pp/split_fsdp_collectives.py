# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import copy, deepcopy
from typing import Literal

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.fx._lazy_graph_module import _make_graph_module

from torchtitan.experiments.graph_trainer.common_utils import (
    copy_parameter_gradient_fqns,
    is_view_like,
    node_argument,
    node_tensor_meta,
    PARAMETER_GRADIENT_FQNS_META,
    same_tensor_metadata,
    same_unary_chain,
    walk_up_unary_chain,
)
from torchtitan.experiments.graph_trainer.debug_utils import tlparse_log_graph_pass
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
    find_fsdp_unshard_outputs_by_param,
    is_all_gather_into_tensor,
    is_reduce_grad_collective,
    is_wait_tensor,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    allow_fx_graph_extraction_of_side_effectful_ops,
    graph_outputs,
    output_names,
    placeholder_names,
    unique_in_order,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META


def _unary_reduce_grad_chain(
    output: fx.Node,
) -> tuple[fx.Node, tuple[fx.Node, ...]] | None:
    """Walk backward from ``output`` and match a unary reduce-grad chain.

    Example::

        activation = x.sin()                            # not matched
        local_grad = activation @ weight                # boundary
        cast_grad = local_grad.to(torch.float32)        # match
        reduced = reduce_scatter_tensor(cast_grad, ...) # match
        output = wait_tensor(reduced)                   # match
    """
    boundary = find_fsdp_reduce_grad_input(output)
    if boundary is None:
        return None
    # Include the annotated cast so local gradients are added before the cast.
    if boundary.target is torch.ops.aten._to_copy.default:
        if len(boundary.all_input_nodes) != 1:
            return None
        cast_input = boundary.all_input_nodes[0]
        cast_input_value = node_tensor_meta(cast_input)
        cast_output_value = node_tensor_meta(boundary)
        if (
            not boundary.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META)
            or boundary.args != (cast_input,)
            or not isinstance(cast_input_value, torch.Tensor)
            or not isinstance(cast_output_value, torch.Tensor)
            or boundary.kwargs != {"dtype": cast_output_value.dtype}
            or cast_input_value.shape != cast_output_value.shape
            or cast_input_value.stride() != cast_output_value.stride()
            or cast_input_value.device != cast_output_value.device
            or cast_input_value.dtype == cast_output_value.dtype
        ):
            return None
        boundary = cast_input

    reverse_nodes = []
    found_collective = False
    node = output
    while node is not boundary:
        inputs = node.all_input_nodes
        if is_reduce_grad_collective(node):
            found_collective = True
        elif not is_wait_tensor(node) and not is_view_like(node):
            return None
        if len(inputs) != 1:
            return None
        reverse_nodes.append(node)
        node = inputs[0]
    if not found_collective:
        return None
    return boundary, tuple(reversed(reverse_nodes))


def _matching_reduce_grad_chains(
    lhs_boundary: fx.Node,
    lhs_nodes: tuple[fx.Node, ...],
    rhs_boundary: fx.Node,
    rhs_nodes: tuple[fx.Node, ...],
    param_fqns: tuple[str, ...],
    add: fx.Node,
) -> bool:
    """Match two reduce-grad chains that differ only at their boundaries.

    Example::

        lhs_cast = lhs.to(torch.float32)                  # lhs match
        lhs_reduce = reduce_scatter_tensor(lhs_cast, ...) # lhs match
        lhs_output = wait_tensor(lhs_reduce)              # lhs match

        rhs_cast = rhs.to(torch.float32)                  # rhs match
        rhs_reduce = reduce_scatter_tensor(rhs_cast, ...) # rhs match
        rhs_output = wait_tensor(rhs_reduce)              # rhs match

        output = lhs_output + rhs_output                  # not matched
    """

    def comparison_chain_without_trailing_aliases(
        nodes: tuple[fx.Node, ...],
    ) -> tuple[fx.Node, ...] | None:
        alias_chain = walk_up_unary_chain(
            nodes[-1],
            add,
            lambda node: node.target is torch.ops.aten.alias.default,
        )
        if alias_chain is None:
            return None
        return nodes[: len(nodes) - len(alias_chain) + 1]

    matched_lhs_nodes = comparison_chain_without_trailing_aliases(lhs_nodes)
    matched_rhs_nodes = comparison_chain_without_trailing_aliases(rhs_nodes)
    if matched_lhs_nodes is None or matched_rhs_nodes is None:
        return False
    return all(
        node.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META) == param_fqns
        for node in (*matched_lhs_nodes, *matched_rhs_nodes)
    ) and same_unary_chain(
        (lhs_boundary, *matched_lhs_nodes),
        (rhs_boundary, *matched_rhs_nodes),
    )


def _is_gradient_add(node: fx.Node) -> bool:
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    ):
        return False
    return node_argument(node, "alpha", 2, 1) == 1 and len(node.args) >= 2


def _unwrap_gradient_output_aliases(output: object) -> fx.Node | None:
    """Walk backward from ``output`` through exclusive storage aliases.

    Example::

        grad = reduced_0 + reduced_1                   # not matched
        alias_0 = torch.ops.aten.alias.default(grad)   # match
        output = torch.ops.aten.alias.default(alias_0) # match
        consumer = output.view(-1)                     # not matched
    """
    if not isinstance(output, fx.Node):
        return None
    node = output
    while node.target is torch.ops.aten.alias.default:
        if len(node.all_input_nodes) != 1 or len(node.users) != 1:
            return None
        node = node.all_input_nodes[0]
    return node


def _functional_gradient_add_tree(output: fx.Node) -> tuple[fx.Node, ...]:
    """Walk backward from ``output`` and match functional gradient adds.

    Example::

        reduced_0 = reduce_grad(local_grad_0)        # not matched
        reduced_1 = reduce_grad(local_grad_1)        # not matched
        reduced_2 = reduce_grad(local_grad_2)        # not matched
        partial = reduced_0 + reduced_1              # match
        output = partial + reduced_2                 # match
        alias = torch.ops.aten.alias.default(output) # not matched
    """
    ordered = []
    visited = set()

    def visit(node: object) -> None:
        if (
            not isinstance(node, fx.Node)
            or node in visited
            or node.target is not torch.ops.aten.add.Tensor
            or not _is_gradient_add(node)
        ):
            return
        visited.add(node)
        visit(node.args[0])
        visit(node.args[1])
        ordered.append(node)

    visit(output)
    return tuple(ordered)


def _inplace_gradient_add_chain(output: fx.Node) -> tuple[fx.Node, ...]:
    """Walk backward from ``output`` and match an in-place add chain.

    Example::

        reduced_0 = reduce_grad(local_grad_0)        # not matched
        reduced_1 = reduce_grad(local_grad_1)        # not matched
        first = accumulator.add_(reduced_0)          # match
        output = first.add_(reduced_1)               # match
        alias = torch.ops.aten.alias.default(output) # not matched
    """
    reverse_order = []
    node = output
    expected_user: fx.Node | None = None
    while node.target is torch.ops.aten.add_.Tensor and _is_gradient_add(node):
        if len(node.users) != 1 or (
            expected_user is not None and expected_user not in node.users
        ):
            return ()
        lhs, rhs = node.args[:2]
        if (
            not isinstance(lhs, fx.Node)
            or not isinstance(rhs, fx.Node)
            or _is_gradient_add(rhs)
            or not same_tensor_metadata(lhs, rhs)
            or not same_tensor_metadata(node, lhs)
        ):
            return ()
        reverse_order.append(node)
        expected_user = node
        node = lhs
    return tuple(reversed(reverse_order))


def _gradient_add_candidates(
    output: object,
) -> tuple[tuple[fx.Node, tuple[str, ...]], ...]:
    """Walk backward from a gradient output and match safe adds.

    Example::

        reduced_0 = reduce_grad(local_grad_0)       # not matched
        reduced_1 = reduce_grad(local_grad_1)       # not matched
        reduced_2 = reduce_grad(local_grad_2)       # not matched
        partial = reduced_0 + reduced_1             # match
        grad = partial + reduced_2                  # match
        output = torch.ops.aten.alias.default(grad) # not matched
    """
    root = _unwrap_gradient_output_aliases(output)
    if root is None:
        return ()
    output_fqns = (
        output.meta.get("custom", {}).get(PARAMETER_GRADIENT_FQNS_META, ())
        if isinstance(output, fx.Node)
        else ()
    )
    adds = (
        _inplace_gradient_add_chain(root)
        if root.target is torch.ops.aten.add_.Tensor
        else _functional_gradient_add_tree(root)
    )
    return tuple(
        (
            add,
            output_fqns
            or add.meta.get("custom", {}).get(PARAMETER_GRADIENT_FQNS_META, ()),
        )
        for add in adds
    )


def _chain_has_exclusive_users(
    nodes: tuple[fx.Node, ...],
    add: fx.Node,
) -> bool:
    """Match a chain used only by its next node and final add.

    Example::

        local_grad = x @ weight                          # not matched
        reduced = reduce_scatter_tensor(local_grad, ...) # match
        waited = wait_tensor(reduced)                    # match
        add = waited + other                             # not in chain
    """
    for index, node in enumerate(nodes):
        expected_user = nodes[index + 1] if index + 1 < len(nodes) else add
        if len(node.users) != 1 or expected_user not in node.users:
            return False
    return True


def _coalesce_fsdp_reduce_grad_add(
    graph: fx.Graph,
    grad_outputs: tuple[object, ...],
) -> None:
    """Move an add after matching reduce-grad chains before one chain.

    Example::

        local_grad_0 = x @ weight_0                           # not matched
        local_grad_1 = x @ weight_1                           # not matched
        reduced_0 = reduce_scatter_tensor(local_grad_0, ...)  # match chain
        reduced_1 = reduce_scatter_tensor(local_grad_1, ...)  # match chain
        grad_0 = wait_tensor(reduced_0)                       # match chain
        grad_1 = wait_tensor(reduced_1)                       # match chain
        output = grad_0 + grad_1                              # match add
        consumer = output.view(-1)                            # not matched

    Output::
        combined = local_grad_0 + local_grad_1
        reduced = reduce_scatter_tensor(combined, ...)
        output = wait_tensor(reduced)
        consumer = output.view(-1)
    """
    current_grad_outputs = list(grad_outputs)
    while True:
        changed = False
        node_order = {node: index for index, node in enumerate(graph.nodes)}
        adds = tuple(
            candidate
            for grad_output in current_grad_outputs
            for candidate in _gradient_add_candidates(grad_output)
        )
        for add, param_fqns in adds:
            if not isinstance(param_fqns, tuple) or len(param_fqns) != 1:
                continue
            lhs, rhs = add.args[:2]
            if (
                not isinstance(lhs, fx.Node)
                or not isinstance(rhs, fx.Node)
                or lhs is rhs
            ):
                continue
            lhs_chain = _unary_reduce_grad_chain(lhs)
            rhs_chain = _unary_reduce_grad_chain(rhs)
            if lhs_chain is None or rhs_chain is None:
                continue
            lhs_boundary, lhs_nodes = lhs_chain
            rhs_boundary, rhs_nodes = rhs_chain
            if not _matching_reduce_grad_chains(
                lhs_boundary,
                lhs_nodes,
                rhs_boundary,
                rhs_nodes,
                param_fqns,
                add,
            ):
                continue
            if not _chain_has_exclusive_users(
                lhs_nodes, add
            ) or not _chain_has_exclusive_users(rhs_nodes, add):
                continue

            if node_order[lhs_boundary] < node_order[rhs_boundary]:
                kept_output, kept_boundary, kept_nodes = (
                    rhs,
                    rhs_boundary,
                    rhs_nodes,
                )
                dropped_nodes = lhs_nodes
            else:
                kept_output, kept_boundary, kept_nodes = (
                    lhs,
                    lhs_boundary,
                    lhs_nodes,
                )
                dropped_nodes = rhs_nodes

            # The later boundary is after both local gradients, so inserting
            # before its chain preserves FX topological order.
            with graph.inserting_before(kept_nodes[0]):
                combined = graph.call_function(
                    torch.ops.aten.add.Tensor,
                    args=(lhs_boundary, rhs_boundary),
                )
            combined.meta = copy(lhs_boundary.meta)
            combined.meta["custom"] = copy(combined.meta.get("custom", {}))
            combined.meta["custom"][PARAMETER_GRADIENT_FQNS_META] = param_fqns
            kept_nodes[0].replace_input_with(kept_boundary, combined)
            kept_output.meta.setdefault("custom", {}).update(add.meta.get("custom", {}))
            add.replace_all_uses_with(kept_output)
            current_grad_outputs[:] = [
                kept_output if output is add else output
                for output in current_grad_outputs
            ]
            # Validate the rewired graph before deleting the mutation chain.
            graph.lint()
            graph.erase_node(add)
            for node in reversed(dropped_nodes):
                assert not node.users
                graph.erase_node(node)
            graph.lint()
            changed = True
            break
        if not changed:
            return


def coalesce_fsdp_reduce_grad_add_pass(
    graph_module: fx.GraphModule,
    _example_inputs: tuple,
) -> fx.GraphModule:
    """Canonicalize repeated-parameter FSDP reductions before scheduling.

    This must run before any GraphTrainer execution-path decision. Direct SPMD
    execution keeps collectives in the joint graph, while GraphPP may split or
    keep them later; both paths require eager FSDP's add-then-reduce ordering.
    """
    grad_outputs = tuple(
        output
        for output in graph_outputs(graph_module.graph)
        if isinstance(output, fx.Node)
        and output.meta.get("custom", {}).get(PARAMETER_GRADIENT_FQNS_META)
    )
    if not grad_outputs:
        return graph_module

    _coalesce_fsdp_reduce_grad_add(graph_module.graph, grad_outputs)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


FSDPExtractionMode = Literal["keep", "cut", "split"]
"""How an FSDP extractor treats the collectives it matches.

- ``keep``: leave them in the compute graph; the input graph is returned.
- ``cut``: remove them from the compute graph only; the collective module is
  ``None``.
- ``split``: remove them from the compute graph and return them as a separate
  collective module.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class FSDPUnshardExtraction:
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
class FSDPReduceGradExtraction:
    """Graph extraction result for FSDP/DDP/HSDP gradient reduction.

    Attributes:
        compute_module (fx.GraphModule): Input graph with reduce-grad
            epilogues removed from parameter-gradient outputs.
        reduce_grad_module (fx.GraphModule | None): Graph that performs
            reduce-scatter/all-reduce epilogues for parameter gradients or the
            FSDP1 persistent-gradient cast, or ``None`` when neither exists.
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
    mode: FSDPExtractionMode = "split",
) -> FSDPUnshardExtraction:
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
        mode (FSDPExtractionMode): ``keep`` returns the input graph,
            ``cut`` removes the all-gather chains from the compute graph, and
            ``split`` also returns them as ``unshard_module``.

    Returns:
        FSDPUnshardExtraction: Extracted modules and calling-convention
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
    if mode == "keep":
        return FSDPUnshardExtraction(
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
                "FSDP extraction expects one unshard chain per flat "
                f"parameter placeholder after deduplication, but "
                f"{param_input.name} has {len(param_unshard_outputs)}. "
                "Run deduplicate_fsdp_unshard_chains_pass before extraction."
            )
        unshard_output = param_unshard_outputs[0]
        found_collective = True
        unshard_outputs.append(unshard_output)

    if not found_collective:
        tlparse_log_graph_pass(graph_module, graph_name="fsdp_compute_no_unshard")
        return FSDPUnshardExtraction(
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
        # Extracted even in "cut" mode: the compute graph's DCE below needs this
        # graph's node set, which includes impure nodes computable from the
        # parameters (waits, prefetch launches) that no unshard output uses.
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

    # The unshard graph's outputs are compute_param_inputs, under the same names.
    unshard_output_names = tuple(node.name for node in compute_param_inputs)
    unshard_module: fx.GraphModule | None = None
    if mode == "split":
        # Extraction preserves mutation-only backward prefetch launches. They
        # have no wait in the unshard graph and must not run as part of UNSHARD.
        _remove_dead_all_gather_launches(unshard_graph)
        unshard_graph.lint()
        unshard_module = _make_graph_module(graph_module, unshard_graph)
        tlparse_log_graph_pass(unshard_module, graph_name="fsdp_unshard")
    compute_module = _make_graph_module(graph_module, compute_graph)
    tlparse_log_graph_pass(compute_module, graph_name="fsdp_compute_no_unshard")
    return FSDPUnshardExtraction(
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
    mode: FSDPExtractionMode = "split",
) -> FSDPReduceGradExtraction:
    """Extract FSDP/DDP/HSDP reduce-grad epilogues from a graph.

    Contract:
      compute(original_inputs)
        -> leading_outputs, reduce_grad_inputs, trailing_outputs

      reduce_grad(unique_reduce_grad_inputs)
        -> original_param_grad_outputs

    Parameter-gradient outputs begin at ``param_grad_output_start``, permitting
    both backward-only graphs and joint graphs with leading outputs such as
    loss. Parameter-gradient slots that do not end in a reduce-scatter,
    all-reduce, or FSDP1 persistent-gradient cast, including ``None`` slots for
    unused or non-differentiable params, are kept in place to preserve the
    one-output-per-param-grad calling convention.

    NOTE: The pre-reduce dtype cast remains in ``compute`` for collective paths.
    For FSDP1, the persistent-gradient cast is extracted so local gradients
    accumulate in the reduce dtype and are cast once later.

    FSDP1 example::

        Input::
            persistent_grad = local_grad.to(torch.float32)

        Output::
            # Compute graph
            compute_output = local_grad

            # Reduce-grad graph
            persistent_grad = compute_output.to(torch.float32)

    Args:
        graph_module (fx.GraphModule): Graph containing parameter-gradient
            outputs.
        num_param_grads (int): Number of parameter-gradient output slots.
        param_grad_output_start (int): Index of the first parameter-gradient
            output. Defaults to zero for backward-only graphs.
        mode (FSDPExtractionMode): ``keep`` returns the input graph (still
            reporting ``reduction_node_names``), ``cut`` removes the reduction
            chains from the compute graph, and ``split`` also returns them as
            ``reduce_grad_module``.

    Returns:
        FSDPReduceGradExtraction: Extracted modules and
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
    found_reduction = False
    for grad_output in grad_outputs:
        reduce_grad_input = find_fsdp_reduce_grad_input(grad_output)
        if reduce_grad_input is not None:
            assert isinstance(grad_output, fx.Node)
            # For example, before extraction of fsdp reduce grad chain:
            #
            #   cast_grad = grad.to(reduce_dtype)
            #   chunks = torch.split(cast_grad, ...)
            #   padded = torch.nn.functional.pad(chunks[-1], ...)
            #   packed_grad = torch.cat((*chunks[:-1], padded))
            #   grad_output = wait_tensor(reduce_scatter_tensor(packed_grad, ...))
            #   grad_output.meta["parameter_gradient_fqns"] = ("weight",)
            #
            # Everything after cast_grad is extracted, to be executed only once.
            # Copy meta parameter_gradient_fqns for further wgrad fusion matching.
            #
            #   cast_grad.meta["parameter_gradient_fqns"] = ("weight",)
            copy_parameter_gradient_fqns(grad_output, reduce_grad_input)
            found_reduction = True
            reduction_outputs.append((grad_output, frozenset((reduce_grad_input,))))
            reduce_grad_inputs.append(reduce_grad_input)
        else:
            reduce_grad_inputs.append(grad_output)

    if not found_reduction:
        tlparse_log_graph_pass(graph_module, graph_name="fsdp_compute_no_reduce_grad")
        return FSDPReduceGradExtraction(
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

    if mode == "keep":
        kept_module = _make_graph_module(graph_module, graph)
        tlparse_log_graph_pass(kept_module, graph_name="fsdp_compute_keep_reduce_grad")
        return FSDPReduceGradExtraction(
            compute_module=kept_module,
            reduce_grad_module=None,
            compute_output_names=output_names(kept_module),
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
        reduce_grad_graph = (
            _extract_graph_with_inputs_outputs(
                graph,
                unique_reduce_grad_inputs,
                list(grad_outputs),
                grad_output_descs,
                "reduce_grad",
                ignore_must_be_in_fw_bw=True,
            )
            if mode == "split"
            else None
        )

    # FX preserves mutation-only tails during DCE. Remove the reduction tail
    # after its inputs become explicit outputs of the backward graph.
    for node in reversed(list(compute_graph.nodes)):
        if node.name in reduction_node_names and not node.users:
            compute_graph.erase_node(node)
    _remove_dead_all_gather_launches(compute_graph)
    compute_graph.lint()

    compute_module = _make_graph_module(graph_module, compute_graph)
    tlparse_log_graph_pass(compute_module, graph_name="fsdp_compute_no_reduce_grad")
    reduce_grad_module: fx.GraphModule | None = None
    if reduce_grad_graph is not None:
        reduce_grad_module = _make_graph_module(graph_module, reduce_grad_graph)
        tlparse_log_graph_pass(reduce_grad_module, graph_name="fsdp_reduce_grad")
    return FSDPReduceGradExtraction(
        compute_module=compute_module,
        reduce_grad_module=reduce_grad_module,
        compute_output_names=output_names(compute_module),
        # The reduction graph's placeholders are the unique reduction inputs.
        reduce_grad_input_names=tuple(node.name for node in unique_reduce_grad_inputs),
        reduction_node_names=frozenset(reduction_node_names),
    )
