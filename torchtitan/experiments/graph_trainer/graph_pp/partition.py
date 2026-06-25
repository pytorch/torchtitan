# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
from dataclasses import dataclass
from typing import Any, Literal

import torch.fx as fx
from torch._functorch.partitioners import _extract_fwd_bwd_outputs

from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    _make_graph_module_like,
    _output_names,
    _placeholder_names,
    extract_graph_with_graph_pp_abi,
)

from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult


GraphPPInputSourceKind = Literal["flat_input", "unsharded_param"]


@dataclass(frozen=True, slots=True)
class GraphPPInputSource:
    """Runtime source for one extracted graph placeholder."""

    name: str
    kind: GraphPPInputSourceKind
    index: int


@dataclass(frozen=True, slots=True)
class GraphMeta:
    """Calling convention for GraphTrainer-native GraphPP fwd/bwd graphs."""

    fwd_input_sources: tuple[GraphPPInputSource, ...]
    num_fwd_user_outputs: int
    saved_for_backward_names: tuple[str, ...]
    fwd_side_effect_output_names: tuple[str, ...]
    bwd_runtime_input_names: tuple[str, ...]
    bwd_runtime_input_indices: tuple[int, ...]
    bwd_input_names: tuple[str, ...]
    bwd_output_names: tuple[str, ...]

    @property
    def num_fwd_inputs(self) -> int:
        return len(self.fwd_input_sources)

    @property
    def num_saved_for_backward(self) -> int:
        return len(self.saved_for_backward_names)

    @property
    def num_fwd_side_effect_outputs(self) -> int:
        return len(self.fwd_side_effect_output_names)

    @property
    def num_bwd_inputs(self) -> int:
        return len(self.bwd_input_names)

    @property
    def num_bwd_runtime_inputs(self) -> int:
        return len(self.bwd_runtime_input_names)

    @property
    def num_bwd_outputs(self) -> int:
        return len(self.bwd_output_names)


def _node_closure(values: list[Any]) -> set[fx.Node]:
    closure: set[fx.Node] = set()
    queue = [value for value in values if isinstance(value, fx.Node)]
    while queue:
        node = queue.pop()
        if node in closure:
            continue
        closure.add(node)
        queue.extend(node.all_input_nodes)
    return closure


def _placeholder_dependencies(values: list[Any]) -> set[fx.Node]:
    return {node for node in _node_closure(values) if node.op == "placeholder"}


def _flat_input_sources(
    gm: fx.GraphModule,
    placeholder_index_by_name: dict[str, int],
) -> tuple[GraphPPInputSource, ...]:
    sources: list[GraphPPInputSource] = []
    for node in gm.graph.find_nodes(op="placeholder"):
        if node.name not in placeholder_index_by_name:
            raise ValueError(
                "Extracted graph placeholder does not come from the traced "
                f"GraphTrainer flat input list: {node.name}"
            )
        sources.append(
            GraphPPInputSource(
                name=node.name,
                kind="flat_input",
                index=placeholder_index_by_name[node.name],
            )
        )
    return tuple(sources)


def _saved_values_for_backward(
    joint: fx.GraphModule,
    *,
    fwd_outputs: list[Any],
    bwd_outputs: list[Any],
    backward_only_names: set[str],
) -> list[fx.Node]:
    """Select forward values consumed only by the extracted backward graph.

    GraphPP uses the stage trace's concrete output ordering instead of
    AOTAutograd's default partition descriptor path. A value is saved when it is
    in the forward dependency closure and has a backward-only user.
    """
    forward_nodes = _node_closure(fwd_outputs)
    backward_nodes = _node_closure(bwd_outputs)
    saved: list[fx.Node] = []
    for node in joint.graph.nodes:
        if node not in forward_nodes or node.name in backward_only_names:
            continue
        if any(
            user in backward_nodes and user not in forward_nodes for user in node.users
        ):
            saved.append(node)
    return saved


def _node_order(joint: fx.GraphModule) -> dict[fx.Node, int]:
    return {node: index for index, node in enumerate(joint.graph.nodes)}


def _is_mutation_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return False
    return any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in schema.arguments
    )


def _alias_base(value: Any) -> fx.Node | None:
    if not isinstance(value, fx.Node):
        return None
    node = value
    while (
        node.op == "call_function"
        and hasattr(node.target, "is_view")
        and node.target.is_view
        and node.args
        and isinstance(node.args[0], fx.Node)
    ):
        node = node.args[0]
    return node


def _forward_side_effects(
    joint: fx.GraphModule,
    *,
    fwd_outputs: list[Any],
    bwd_outputs: list[Any],
    backward_only_names: set[str],
) -> tuple[list[fx.Node], list[fx.Node]]:
    """Find forward-side mutations that must execute in the forward callable.

    Chunked loss traces populate the hidden-state gradient accumulator with
    ``copy_`` side effects during the loss forward. The accumulator is later
    consumed by decoder backward, but those writes are not visible in the data
    dependency closure of the loss output. Preserve the mutated base as a saved
    value and keep the mutation nodes as forward-only outputs so extracted
    forward graph execution materializes the post-mutation tensor.

    Other forward mutations, such as MoE expert-count buffer updates, are not
    consumed by backward but still affect training state. Keep those mutation
    nodes as forward-only outputs as well; the runtime ignores the extra values,
    but returning them keeps the side effects live in the extracted graph.
    """

    order = _node_order(joint)
    fwd_output_nodes = [value for value in fwd_outputs if isinstance(value, fx.Node)]
    if not fwd_output_nodes:
        return [], []
    fwd_output_barrier = max(order[node] for node in fwd_output_nodes)
    backward_nodes = _node_closure(bwd_outputs)
    saved_bases: list[fx.Node] = []
    side_effect_outputs: list[fx.Node] = []

    for node in joint.graph.nodes:
        if (
            order[node] > fwd_output_barrier
            or node in backward_nodes
            or not _is_mutation_node(node)
        ):
            continue
        if not node.args:
            continue
        base = _alias_base(node.args[0])
        if base is None or base.name in backward_only_names:
            continue
        side_effect_outputs.append(node)
        if base.op != "placeholder" and base in backward_nodes:
            saved_bases.append(base)

    return (
        list(dict.fromkeys(saved_bases).keys()),
        list(dict.fromkeys(side_effect_outputs).keys()),
    )


def _backward_passthrough_placeholders(
    *,
    bwd_outputs: list[Any],
    backward_only_names: set[str],
) -> list[fx.Node]:
    """Preserve flat ABI placeholders that are returned by backward directly.

    minimal_fx_tracer unwraps tensor subclasses into plain graph values. A
    DTensor gradient, for example, may flatten to ``(local_grad, device_mesh)``,
    where ``device_mesh`` is an input placeholder carried through only so the
    runtime can rewrap the output subclass. Since such placeholders have no
    compute users, dependency-based saved-value discovery will not select them;
    they still must be available to the extracted backward graph.
    """
    return list(
        dict.fromkeys(
            node
            for node in bwd_outputs
            if isinstance(node, fx.Node)
            and node.op == "placeholder"
            and node.name not in backward_only_names
        )
    )


def _is_getitem_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _is_tuple_like_node(node: fx.Node) -> bool:
    return isinstance(node.meta.get("val"), (tuple, list))


def _expand_tuple_saved_value(
    node: fx.Node,
    *,
    backward_nodes: set[fx.Node],
    order: dict[fx.Node, int],
) -> list[fx.Node]:
    if not _is_tuple_like_node(node):
        return [node]

    backward_users = [user for user in node.users if user in backward_nodes]
    getitem_users = [user for user in backward_users if _is_getitem_node(user)]
    if len(getitem_users) != len(backward_users):
        return [node]
    if not getitem_users:
        return [node]

    expanded: list[fx.Node] = []
    for user in sorted(getitem_users, key=order.__getitem__):
        expanded.extend(
            _expand_tuple_saved_value(
                user,
                backward_nodes=backward_nodes,
                order=order,
            )
        )
    return expanded


def _flatten_tuple_saved_values(
    joint: fx.GraphModule,
    *,
    saved_values: list[fx.Node],
    bwd_outputs: list[Any],
) -> list[fx.Node]:
    """Expose saved tuple intermediates as tensor leaves at the GraphPP ABI.

    Higher-order ops such as FlexAttention can return nested tuples whose tensor
    leaves are consumed by backward. Keeping the raw tuple as a forward output
    works for interpreted FX, but standalone regional Inductor expects compiled
    regions to expose plain tensor outputs. If backward only observes the tuple
    through getitem chains, save those getitem leaves directly.
    """

    backward_nodes = _node_closure(bwd_outputs)
    order = _node_order(joint)
    flattened: list[fx.Node] = []
    for node in saved_values:
        flattened.extend(
            _expand_tuple_saved_value(
                node,
                backward_nodes=backward_nodes,
                order=order,
            )
        )
    return list(dict.fromkeys(flattened))


def partition_joint_graph(
    traced: TracedResult,
    *,
    num_fwd_outputs: int,
    backward_only_input_indices: tuple[int, ...] = (),
) -> tuple[fx.GraphModule, fx.GraphModule, GraphMeta]:
    """Partition a post-pass GraphTrainer joint graph into GraphPP callables.

    Contract:
      ``traced.gm`` is a flat ``minimal_fx_tracer`` graph whose outputs are
      ordered as forward user values followed by parameter gradients and input
      gradients. ``backward_only_input_indices`` names tangent placeholders,
      such as non-last-stage output grads, that must not become forward inputs.
      The returned forward graph emits user outputs, saved values, and ignored
      side-effect materialization outputs; the backward graph consumes saved
      values plus runtime tangents and emits gradients.

      GraphPP owns this partitioning policy. PyTorch extraction utilities are
      used only after this pass has explicitly selected the flat graph ABI.

    Pseudocode:
      split traced outputs into forward outputs and backward outputs
      discover saved values from forward-to-backward dependencies
      preserve passthrough subclass metadata placeholders and forward mutations
      flatten tuple saved values for Inductor-friendly callable outputs
      extract forward graph from forward inputs to user/saved/side-effect outputs
      extract backward graph from saved/runtime tangent inputs to gradients
      return both graphs plus GraphMeta calling-convention metadata

    GraphTrainer traces a pure flat-input graph whose raw outputs are ordered as
    forward user outputs followed by gradients. ``num_fwd_outputs`` is the raw
    tensor ABI count after minimal_fx_tracer unwraps tensor subclasses; semantic
    pytree structure is restored by GraphPPRunner at the runtime boundary.
    The caller is responsible for applying FX-preserving GraphTrainer passes
    before calling this function.
    """
    if num_fwd_outputs < 1:
        raise ValueError(f"num_fwd_outputs must be positive, got {num_fwd_outputs}")

    joint = copy.deepcopy(traced.gm)
    placeholders = list(joint.graph.find_nodes(op="placeholder"))
    placeholder_index_by_name = {
        node.name: index for index, node in enumerate(placeholders)
    }
    backward_only_index_set = set(backward_only_input_indices)
    if any(
        index >= len(placeholders) or index < 0 for index in backward_only_index_set
    ):
        raise ValueError(
            "backward_only_input_indices must reference traced graph placeholders: "
            f"{sorted(backward_only_index_set)} for {len(placeholders)} placeholders"
        )
    backward_only_inputs = [
        node
        for index, node in enumerate(placeholders)
        if index in backward_only_index_set
    ]
    backward_only_names = {node.name for node in backward_only_inputs}

    (
        fwd_outputs,
        bwd_outputs,
        fwd_output_descs,
        bwd_output_descs,
    ) = _extract_fwd_bwd_outputs(joint, num_fwd_outputs=num_fwd_outputs)
    saved_values = _saved_values_for_backward(
        joint,
        fwd_outputs=fwd_outputs,
        bwd_outputs=bwd_outputs,
        backward_only_names=backward_only_names,
    )
    saved_values.extend(
        _backward_passthrough_placeholders(
            bwd_outputs=bwd_outputs,
            backward_only_names=backward_only_names,
        )
    )
    (side_effect_saved_values, fwd_side_effect_outputs,) = _forward_side_effects(
        joint,
        fwd_outputs=fwd_outputs,
        bwd_outputs=bwd_outputs,
        backward_only_names=backward_only_names,
    )
    saved_values = list(dict.fromkeys([*saved_values, *side_effect_saved_values]))
    saved_values = _flatten_tuple_saved_values(
        joint,
        saved_values=saved_values,
        bwd_outputs=bwd_outputs,
    )

    fw_outputs = fwd_outputs + saved_values + fwd_side_effect_outputs
    fw_output_descs = fwd_output_descs + [None] * (
        len(saved_values) + len(fwd_side_effect_outputs)
    )
    fw_input_set = _placeholder_dependencies(fw_outputs)
    fw_inputs = [
        node
        for node in placeholders
        if node in fw_input_set and node.name not in backward_only_names
    ]

    bwd_input_set = _node_closure(bwd_outputs)
    runtime_bwd_inputs = [
        node for node in backward_only_inputs if node in bwd_input_set
    ]
    runtime_bwd_input_indices = tuple(
        index
        for index, node in enumerate(backward_only_inputs)
        if node in bwd_input_set
    )
    bw_inputs = saved_values + runtime_bwd_inputs

    fw_graph = extract_graph_with_graph_pp_abi(
        joint.graph,
        fw_inputs,
        fw_outputs,
        fw_output_descs,
        "forward",
    )
    bw_graph = extract_graph_with_graph_pp_abi(
        joint.graph,
        bw_inputs,
        bwd_outputs,
        bwd_output_descs,
        "backward",
    )
    fw_module = _make_graph_module_like(joint, fw_graph)
    bw_module = _make_graph_module_like(joint, bw_graph)
    fw_module.graph.lint()
    bw_module.graph.lint()
    fw_module.recompile()
    bw_module.recompile()

    return (
        fw_module,
        bw_module,
        GraphMeta(
            fwd_input_sources=_flat_input_sources(
                fw_module,
                placeholder_index_by_name,
            ),
            num_fwd_user_outputs=num_fwd_outputs,
            saved_for_backward_names=_output_names(fw_module)[
                num_fwd_outputs : num_fwd_outputs + len(saved_values)
            ],
            fwd_side_effect_output_names=_output_names(fw_module)[
                num_fwd_outputs + len(saved_values) :
            ],
            bwd_runtime_input_names=tuple(node.name for node in runtime_bwd_inputs),
            bwd_runtime_input_indices=runtime_bwd_input_indices,
            bwd_input_names=_placeholder_names(bw_module),
            bwd_output_names=_output_names(bw_module),
        ),
    )
