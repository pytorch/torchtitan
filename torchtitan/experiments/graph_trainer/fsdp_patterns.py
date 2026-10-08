# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pattern helpers for GraphPP's current simple-FSDP collective traces.

The matchers intentionally follow c10d functional traces produced by FSDP2:

    param_shard -> all_gather -> wait -> view*/split-cat -> compute
    local_grad -> cast/view* -> reduce_scatter -> wait -> param_grad
    local_grad -> cast/view* -> all_reduce -> wait -> param_grad

SimpleFSDP traces annotate the unshard construction with its parameter FQN.
The provenance match runs before collective bucketing and uses that provenance
to stop before real compute. Its annotations keep the discovered parameter
boundary available to later graph passes after the original all-gather and
wait nodes have been replaced.
"""

import operator
from collections.abc import Iterable
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from torchtitan.experiments.graph_trainer.common_utils import (
    _op_arg_by_name,
    dtype_only_to_copy_input,
    is_view_like,
    node_tensor_meta,
    PARAMETER_GRADIENT_FQNS_META,
    sole_user,
    unary_chain_to_boundary,
)
from torchtitan.experiments.graph_trainer.mutation_utils import (
    base_tensor_for_mutation_target,
    mutation_deps,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    FSDP_MESH_AXIS_NAMES_META,
    FSDP_PARAM_FQNS_META,
    FSDP_REDUCE_DTYPE_META,
)


_FSDP_UNSHARD_OUTPUT_PARAM_NAMES = "fsdp_unshard_output_param_names"
_FSDP_UNSHARD_CONSUMER_INPUT_PATHS = "fsdp_unshard_consumer_input_paths"
_FSDP_UNSHARD_ANNOTATED = "fsdp_unshard_annotated"


def is_wait_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_reduce_scatter_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_all_reduce(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_reduce.default
    )


def is_reduce_grad_collective(node: fx.Node) -> bool:
    return is_reduce_scatter_tensor(node) or is_all_reduce(node)


def _fsdp_chain_inputs(node: fx.Node) -> list[fx.Node]:
    """Return chain inputs, ignoring precompile's process-group edge.

    Precompile adds a special op to get the process group, which becomes an
    input to the communication op. For example::

        pg = torch.ops._dtensor.mesh_get_process_group(mesh, dim=0)
        gathered = all_gather_into_tensor(
            input=param,
            group_name=pg,
        )

    FSDP communication detection follows a unary chain that ends in the
    communication op. The extra ``group_name`` input breaks that assumption,
    so ignore it when applying the unary-chain check.
    """
    input_nodes = node.all_input_nodes
    if is_all_gather_into_tensor(node) or is_reduce_grad_collective(node):
        group_name = _op_arg_by_name(node, "group_name")
        input_nodes = [
            input_node for input_node in input_nodes if input_node is not group_name
        ]
    return input_nodes


def fsdp_param_fqns(node: fx.Node) -> tuple[str, ...]:
    """Return the SimpleFSDP parameter provenance attached to a graph node."""
    return node.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META, ())


def _find_last_all_gather_in_chain(start_node: fx.Node) -> fx.Node | None:
    """Find the final all-gather in a linear FSDP unshard launch chain."""
    node = start_node
    last_all_gather = None
    while True:
        if is_all_gather_into_tensor(node):
            if not fsdp_param_fqns(node):
                break
            last_all_gather = node
        if len(node.users) != 1:
            break
        user = next(iter(node.users))
        if len(_fsdp_chain_inputs(user)) > 1:
            break
        node = user
    return last_all_gather


def _find_last_user_in_wait_chain(wait_node: fx.Node) -> fx.Node:
    """Find the last FSDP unshard node before the value enters real compute.

    The traced FSDP unshard has a mostly linear shape:

        flat_param -> ... -> all_gather -> wait -> view* -> compute

    Some models reshape the gathered flat buffer through a split/cat fanout:

        wait -> split -> getitem_0 --+
                      -> getitem_1 --+-> cat -> view* -> compute

    Trace-time parameter metadata defines the region. The split/getitem/cat
    fanout is included when all of its nodes carry the same provenance.
    """
    param_fqns = fsdp_param_fqns(wait_node)
    node = wait_node
    while True:
        users = tuple(
            user
            for user in node.users
            if not user.meta.get("autograd_backward", False)
            and fsdp_param_fqns(user) == param_fqns
        )

        if len(users) != 1:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.Tensor
                and users
                and all(
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.users) == 1
                    for user in users
                )
            ):
                getitem_users = [next(iter(user.users)) for user in users]
                potential_cat = getitem_users[0]
                if all(user == potential_cat for user in getitem_users) and (
                    potential_cat.op == "call_function"
                    and potential_cat.target == torch.ops.aten.cat.default
                    and fsdp_param_fqns(potential_cat) == param_fqns
                ):
                    node = potential_cat
                    continue
            break

        node = users[0]
    return node


def _find_last_non_view_node_in_chain(node: fx.Node) -> fx.Node:
    """Return the value GraphPP should pass across the unshard boundary."""
    result = node
    while hasattr(result.target, "is_view") and result.target.is_view:
        if len(result.all_input_nodes) != 1:
            raise ValueError(f"View node {result.name} should have exactly one input")
        result = result.all_input_nodes[0]
    return result


def _unshard_output_from_all_gather(last_all_gather: fx.Node) -> fx.Node:
    if len(last_all_gather.users) != 1:
        raise ValueError(
            f"Expected one wait_tensor user for all_gather node {last_all_gather.name}, "
            f"got {len(last_all_gather.users)}"
        )
    wait_node = next(iter(last_all_gather.users))
    if not is_wait_tensor(wait_node):
        raise ValueError(
            f"Expected wait_tensor after all_gather node {last_all_gather.name}, "
            f"got {wait_node.name}"
        )

    all_gather_fqns = fsdp_param_fqns(last_all_gather)
    if not all_gather_fqns:
        raise ValueError(
            f"FSDP all-gather node {last_all_gather.name} does not carry "
            "parameter provenance"
        )
    wait_fqns = fsdp_param_fqns(wait_node)
    if all_gather_fqns != wait_fqns:
        raise ValueError(
            "FSDP trace metadata does not match between all-gather "
            f"{last_all_gather.name} {all_gather_fqns} and wait "
            f"{wait_node.name} {wait_fqns}"
        )

    wait_chain_user = _find_last_user_in_wait_chain(wait_node)
    output = _find_last_non_view_node_in_chain(wait_chain_user)
    if fsdp_param_fqns(output) != all_gather_fqns:
        raise ValueError(
            f"FSDP unshard output {output.name} does not carry parameter "
            f"metadata {all_gather_fqns}"
        )
    return output


def _find_fsdp_unshard_outputs(
    param_placeholder: fx.Node,
) -> tuple[fx.Node, ...]:
    """Match FSDP unshard outputs in the original, unbucketed graph."""
    last_all_gather = _find_last_all_gather_in_chain(param_placeholder)
    if last_all_gather is not None:
        return (_unshard_output_from_all_gather(last_all_gather),)

    outputs: list[fx.Node] = []
    seen: set[fx.Node] = set()
    for user in param_placeholder.users:
        if len(_fsdp_chain_inputs(user)) > 1:
            continue
        last_all_gather = _find_last_all_gather_in_chain(user)
        if last_all_gather is None:
            continue
        output = _unshard_output_from_all_gather(last_all_gather)
        if output not in seen:
            seen.add(output)
            outputs.append(output)
    return tuple(outputs)


def annotate_fsdp_unshard_outputs(gm: fx.GraphModule) -> None:
    """Preserve FSDP parameter boundaries across collective bucketing.

    Bucketing replaces each original all-gather and wait with a shared bucket
    plus reconstructed parameter values. Record both the selected unshard
    output and its consumer input edges before that rewrite. Parameter
    reconstruction after the wait, including weight quantization, may remain
    unchanged and keep the output marker. If the marked output is replaced,
    its recorded consumer edges identify the replacement value.
    """
    for placeholder in gm.graph.find_nodes(op="placeholder"):
        outputs = _find_fsdp_unshard_outputs(placeholder)
        if not outputs:
            continue
        placeholder.meta[_FSDP_UNSHARD_ANNOTATED] = True
        for output in outputs:
            output_param_names = tuple(
                output.meta.get(_FSDP_UNSHARD_OUTPUT_PARAM_NAMES, ())
            )
            if placeholder.name not in output_param_names:
                output.meta[_FSDP_UNSHARD_OUTPUT_PARAM_NAMES] = (
                    *output_param_names,
                    placeholder.name,
                )

            for consumer in output.users:
                flat_inputs, _ = pytree.tree_flatten_with_path(
                    (consumer.args, consumer.kwargs)
                )
                input_paths = tuple(
                    path for path, input_node in flat_inputs if input_node is output
                )
                if not input_paths:
                    continue
                consumer_inputs = dict(
                    consumer.meta.get(_FSDP_UNSHARD_CONSUMER_INPUT_PATHS, {})
                )
                consumer_inputs[placeholder.name] = input_paths
                consumer.meta[_FSDP_UNSHARD_CONSUMER_INPUT_PATHS] = consumer_inputs


def _depends_on(
    node: fx.Node,
    ancestor: fx.Node,
    mutation_writers: dict[fx.Node, list[fx.Node]],
    node_order: dict[fx.Node, int],
) -> bool:
    pending = [node]
    seen: set[fx.Node] = set()
    while pending:
        candidate = pending.pop()
        if candidate is ancestor:
            return True
        if candidate in seen:
            continue
        seen.add(candidate)
        pending.extend(candidate.all_input_nodes)
        pending.extend(
            writer
            for writer in mutation_writers.get(
                base_tensor_for_mutation_target(candidate), ()
            )
            if node_order[writer] < node_order[node]
        )
    return False


def find_fsdp_unshard_outputs_by_param(
    param_placeholders: Iterable[fx.Node],
) -> dict[fx.Node, tuple[fx.Node, ...]]:
    """Find FSDP unshard outputs for multiple parameters with one graph scan."""
    placeholders = tuple(param_placeholders)
    if not placeholders:
        return {}

    graph = placeholders[0].graph
    if any(placeholder.graph is not graph for placeholder in placeholders):
        raise ValueError("FSDP parameter placeholders must belong to one graph")

    outputs_by_param = {
        placeholder: _find_fsdp_unshard_outputs(placeholder)
        for placeholder in placeholders
        if not placeholder.meta.get(_FSDP_UNSHARD_ANNOTATED, False)
    }
    annotated_params = {
        placeholder.name: placeholder
        for placeholder in placeholders
        if placeholder.meta.get(_FSDP_UNSHARD_ANNOTATED, False)
    }
    if not annotated_params:
        return outputs_by_param

    mutation_writers = mutation_deps(graph)
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    marked_outputs: dict[str, list[fx.Node]] = {name: [] for name in annotated_params}
    consumer_paths: dict[str, list[tuple[fx.Node, pytree.KeyPath]]] = {
        name: [] for name in annotated_params
    }
    for node in graph.nodes:
        if node.op == "placeholder" or node.meta.get("autograd_backward", False):
            continue
        for param_name in node.meta.get(_FSDP_UNSHARD_OUTPUT_PARAM_NAMES, ()):
            param = annotated_params.get(param_name)
            if param is not None and _depends_on(
                node, param, mutation_writers, node_order
            ):
                marked_outputs[param_name].append(node)
        for param_name, paths in node.meta.get(
            _FSDP_UNSHARD_CONSUMER_INPUT_PATHS, {}
        ).items():
            if param_name in annotated_params:
                consumer_paths[param_name].extend((node, path) for path in paths)

    for param_name, param_placeholder in annotated_params.items():
        if marked_outputs[param_name]:
            outputs_by_param[param_placeholder] = tuple(marked_outputs[param_name])
            continue

        outputs: list[fx.Node] = []
        seen: set[fx.Node] = set()
        for consumer, input_path in consumer_paths[param_name]:
            try:
                output = pytree.key_get((consumer.args, consumer.kwargs), input_path)
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"FSDP unshard consumer {consumer.name} lost input path "
                    f"{pytree.keystr(input_path)} for parameter {param_name}"
                ) from exc
            if not isinstance(output, fx.Node):
                raise ValueError(
                    f"FSDP unshard consumer {consumer.name} input path "
                    f"{pytree.keystr(input_path)} for parameter {param_name} "
                    "no longer resolves to an FX node"
                )
            if not _depends_on(output, param_placeholder, mutation_writers, node_order):
                raise ValueError(
                    f"FSDP unshard consumer {consumer.name} input path "
                    f"{pytree.keystr(input_path)} no longer depends on parameter "
                    f"{param_name}"
                )
            if output not in seen:
                seen.add(output)
                outputs.append(output)
        if not outputs and param_placeholder.users:
            raise ValueError(
                f"FSDP parameter {param_name} lost its annotated unshard output"
            )
        outputs_by_param[param_placeholder] = tuple(outputs)

    return outputs_by_param


def find_fsdp_unshard_outputs(param_placeholder: fx.Node) -> tuple[fx.Node, ...]:
    """Return all FSDP unshard outputs launched from one flat parameter input.

    Most parameters have one linear all-gather chain. Some real traces read the
    same parametrized value more than once, which produces multiple equivalent
    all-gather/wait chains from the same placeholder.
    ``deduplicate_fsdp_unshard_chains_pass`` canonicalizes those duplicate
    chains and annotates their boundaries before downstream FSDP passes rely on
    a single unsharded value. The annotation remains valid after FSDP bucketing
    replaces the original collective and wait nodes.
    """
    return find_fsdp_unshard_outputs_by_param((param_placeholder,))[param_placeholder]


def find_fsdp_unshard_output(param_placeholder: fx.Node) -> fx.Node | None:
    """Return the extracted unshard output for one flat parameter input.

    GraphPP and the SAC force-save policy must agree on this node. It is the
    same value AutoParallel saves for ``reshard_after_forward=False``: the last
    non-view node in the FSDP all-gather/wait reconstruction chain. Parameters
    without an all-gather are replicated or otherwise already local, so callers
    should keep the original placeholder as that parameter's unsharded value.

    """
    outputs = find_fsdp_unshard_outputs(param_placeholder)
    if not outputs:
        return None
    return outputs[0]


def find_fsdp_unshard_save_node(param_placeholder: fx.Node) -> fx.Node | None:
    return find_fsdp_unshard_output(param_placeholder)


def find_fsdp_unshard_save_nodes(param_placeholder: fx.Node) -> tuple[fx.Node, ...]:
    """Return all FSDP unshard values that SAC must save for one parameter."""
    return find_fsdp_unshard_outputs(param_placeholder)


def find_fsdp_reduce_grad_collective_chain(
    param_grad_output: Any,
) -> tuple[fx.Node, tuple[fx.Node, ...]] | None:
    """Find the earliest reduce-grad collective input and its unary suffix.

    Example::

        local_grad = x @ weight                         # not matched
        cast_grad = local_grad.to(torch.float32)        # boundary
        reduced = reduce_scatter_tensor(cast_grad, ...) # match
        output = wait_tensor(reduced)                   # match
    """
    if not isinstance(param_grad_output, fx.Node):
        return None

    node = param_grad_output
    reverse_nodes: list[fx.Node] = []
    matched: tuple[fx.Node, tuple[fx.Node, ...]] | None = None
    while True:
        input_nodes = _fsdp_chain_inputs(node)
        if len(input_nodes) != 1:
            break
        input_node = input_nodes[0]
        if not sole_user(input_node, node):
            break
        reverse_nodes.append(node)
        if is_reduce_grad_collective(node):
            matched = (input_node, tuple(reversed(reverse_nodes)))
        node = input_node
    return matched


def find_fsdp_reduce_grad_input(
    param_grad_output: Any,
    *,
    collective_param_fqns: frozenset[tuple[str, ...]] | None = None,
) -> fx.Node | None:
    """Return the split point before an FSDP reduce-grad epilogue.

    The backward FSDP/DDP/HSDP tail is traced as a unary chain ending in the
    parameter-grad output:

        local_grad -> cast/view* -> reduce_scatter -> wait -> sharded_grad
        local_grad -> cast/view* -> all_reduce -> wait -> replicated_grad
        local_grad -> cast/view* -> all_reduce -> wait -> reduce_scatter
          -> wait -> grad
        local_grad -> persistent-dtype cast -> grad  # FSDP1

    GraphPP extracts at the input to the earliest grad-sync collective in that
    suffix. For an annotated SimpleFSDP layout, it extracts at the layout input
    instead. For FSDP1, it extracts before the persistent-dtype cast. The
    pre-reduce cast remains in the compute graph so microbatch accumulation
    happens in FSDP's reduce dtype. Values that are not FX nodes, such as
    ``None`` parameter-grad slots, are preserved by the caller.
    """
    matched = find_fsdp_reduce_grad_collective_chain(param_grad_output)
    # The collective scan identifies ``packed_grad`` in code shaped like:
    #
    #   cast_grad = grad.to(reduce_dtype)
    #   chunks = torch.split(cast_grad, ...)
    #   padded = torch.nn.functional.pad(chunks[-1], ...)
    #   packed_grad = torch.cat((*chunks[:-1], padded))
    #   reduced_grad = reduce_scatter_tensor(packed_grad, ...)
    #
    # reduce_grad_input is packed_grad.
    # None means the scan found no reduce-gradient collective.
    if matched is None:
        return (
            _find_persistent_grad_cast_input(
                param_grad_output,
                collective_param_fqns=collective_param_fqns,
            )
            if isinstance(param_grad_output, fx.Node)
            else None
        )
    reduce_grad_input, _ = matched
    # For an annotated layout, look back to cast_grad:
    #
    #   # Repeated schedule action
    #   cast_grad = grad.to(reduce_dtype)
    #   grad_accumulator.add_(cast_grad)
    #
    #   # Final reduce_grad
    #   chunks = torch.split(grad_accumulator, ...)
    #   padded = torch.nn.functional.pad(chunks[-1], ...)
    #   packed_grad = torch.cat((*chunks[:-1], padded))
    #   reduced_grad = reduce_scatter_tensor(packed_grad, ...)
    return _find_grad_compute_boundary(reduce_grad_input)


def find_fsdp_unary_reduce_grad_chain(
    param_grad_output: Any,
) -> tuple[fx.Node, tuple[fx.Node, ...]] | None:
    """Find a unary reduce-grad chain and its compute boundary.

    Example::

        activation = x.sin()                            # not matched
        local_grad = activation @ weight                # boundary
        cast_grad = local_grad.to(torch.float32)        # match
        reduced = reduce_scatter_tensor(cast_grad, ...) # match
        output = wait_tensor(reduced)                   # match
    """
    matched = find_fsdp_reduce_grad_collective_chain(param_grad_output)
    if matched is None:
        return None
    collective_input, nodes = matched
    boundary = _find_grad_compute_boundary(collective_input)

    layout_nodes = unary_chain_to_boundary(collective_input, boundary, is_view_like)
    if layout_nodes is None:
        return None
    nodes = (*layout_nodes, *nodes)

    if boundary.target is torch.ops.aten._to_copy.default:
        cast = boundary
        cast_input = dtype_only_to_copy_input(cast)
        if (
            not boundary.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META)
            or cast_input is None
        ):
            return None
        boundary = cast_input
        nodes = (cast, *nodes)

    if any(
        not is_reduce_grad_collective(node)
        and not is_wait_tensor(node)
        and not is_view_like(node)
        for node in nodes
    ):
        return None
    return boundary, nodes


def _parameter_gradient_matches_fsdp_parameter(
    *,
    fsdp_param_fqn: str,
    parameter_grad_fqns: tuple[str, ...],
    module_fqn: object,
) -> bool:
    if fsdp_param_fqn in parameter_grad_fqns:
        return True
    return isinstance(module_fqn, str) and (
        f"{module_fqn}.{fsdp_param_fqn}" in parameter_grad_fqns
    )


def _find_persistent_grad_cast_input(
    param_grad_output: fx.Node,
    collective_param_fqns: frozenset[tuple[str, ...]] | None,
) -> fx.Node | None:
    """Return an FSDP1 gradient before its persistent-dtype cast."""

    custom = param_grad_output.meta.get("custom", {})
    cast_input = dtype_only_to_copy_input(param_grad_output)
    if (
        FSDP_REDUCE_DTYPE_META not in custom
        or cast_input is None
        or len(param_grad_output.users) != 1
        or next(iter(param_grad_output.users)).op != "output"
    ):
        return None

    if collective_param_fqns is None:
        collective_param_fqns = frozenset(
            fqns
            for node in param_grad_output.graph.nodes
            if is_reduce_grad_collective(node) and (fqns := fsdp_param_fqns(node))
        )

    input_value = node_tensor_meta(cast_input)
    reduce_dtype = custom[FSDP_REDUCE_DTYPE_META]
    output_param_fqns = custom.get(FSDP_PARAM_FQNS_META, ())
    parameter_grad_fqns = custom.get(PARAMETER_GRADIENT_FQNS_META, ())
    mesh_axis_names = custom.get(FSDP_MESH_AXIS_NAMES_META, ())
    module_fqn = custom.get("module_fqn")
    if (
        input_value is None
        or not isinstance(reduce_dtype, torch.dtype)
        or input_value.dtype != reduce_dtype
        or len(output_param_fqns) != 1
        or not parameter_grad_fqns
        or not mesh_axis_names
        or output_param_fqns in collective_param_fqns
        or not _parameter_gradient_matches_fsdp_parameter(
            fsdp_param_fqn=output_param_fqns[0],
            parameter_grad_fqns=parameter_grad_fqns,
            module_fqn=module_fqn,
        )
    ):
        return None
    return cast_input


def _find_grad_compute_boundary(collective_input: fx.Node) -> fx.Node:
    """Find the value to accumulate before the SimpleFSDP layout.

    For example, given::

        cast_grad = grad.to(reduce_dtype)
        chunks = torch.split(cast_grad, ...)
        padded = torch.nn.functional.pad(chunks[-1], ...)
        collective_input = torch.cat((*chunks[:-1], padded))

    return ``cast_grad``.
    Without the cast, return ``grad``.
    """
    param_fqns = fsdp_param_fqns(collective_input)
    if not param_fqns:
        return collective_input

    layout_nodes: set[fx.Node] = set()
    pending = [collective_input]
    while pending:
        node = pending.pop()
        if node in layout_nodes or fsdp_param_fqns(node) != param_fqns:
            continue
        layout_nodes.add(node)
        pending.extend(node.all_input_nodes)

    boundary_nodes = {
        input_node
        for node in layout_nodes
        for input_node in node.all_input_nodes
        if input_node not in layout_nodes
        and isinstance(input_node.meta.get("val"), torch.Tensor)
    }
    if len(boundary_nodes) != 1:
        raise ValueError(
            "Expected one tensor input to the FSDP reduce-grad layout for "
            f"{param_fqns}, found {len(boundary_nodes)}"
        )
    (boundary,) = boundary_nodes

    layout_users = [user for user in boundary.users if user in layout_nodes]
    if (
        len(layout_users) == 1
        and layout_users[0].target is torch.ops.aten._to_copy.default
    ):
        return layout_users[0]
    return boundary
